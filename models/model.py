"""mmFormer-inspired multi-view cardiac MRI segmenter with convolutional 2x2x2 patch embedding.

Architecture adapted from Zhang et al., MICCAI 2022 (mmFormer):
  modality-specific CNN + intra-modal Transformer encoders;
  missing-modality-aware inter-modal Transformer;
  progressive CNN decoder, shared auxiliary encoder decoder and deep supervision.
Extensions: five anatomy tokens, anatomical visibility mask, anatomy-to-patch
fusion and view-specific segmentation heads (2CH/4CH/SA).

This is an independent, runnable adaptation, NOT an exact copy of the official
mmFormer repository. See https://github.com/YaoZhang93/mmFormer.

Input: x [B, 3, D, H, W], modality_mask [B, 3] (1 = available).
Output dict: logits: {view: [B, num_classes, D, H, W]},
             aux_logits, deep_logits, anatomy_tokens, pred_tokens, valid_views.
Only available views should contribute to segmentation loss.
Tokenization: view-specific Conv3d(kernel_size=2, stride=2) on the CNN bottleneck, followed by flatten.
Odd bottleneck dimensions are padded on the right to ensure all voxels contribute.
`token_grid` is a positional-embedding reference grid, not the token count.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

VIEWS = ("2ch", "4ch", "sa")
ANATOMY = ("LV_myo", "LV_cav", "RV_cav", "RA", "LA")
VISIBILITY = ((1,1,1),(1,1,1),(0,1,1),(0,1,0),(0,1,0))


def norm(c: int) -> nn.Module:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(cin, cout, 3, stride, 1, bias=False),
                                  norm(cout), nn.ReLU(inplace=True),
                                  nn.Conv3d(cout, cout, 3, 1, 1, bias=False), norm(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv3d(cin, cout, 1, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4*dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        y = self.n1(x)
        z, _ = self.attn(y, y, y, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + z
        return x + self.mlp(self.n2(x))


class HybridViewEncoder(nn.Module):
    """Independent CNN and intra-modal self-attention per view."""
    def __init__(self, in_channels: int, base: int, dim: int,
                 token_grid: Tuple[int,int,int], intra_layers: int, heads: int):
        super().__init__()
        self.c1 = ConvBlock(in_channels, base)
        self.c2 = ConvBlock(base, base*2, 2)
        self.c3 = ConvBlock(base*2, base*4, 2)
        self.c4 = ConvBlock(base*4, dim, 2)
        self.patch_embed = nn.Conv3d(dim, dim, kernel_size=2, stride=2)
        self.position_grid = tuple(token_grid)
        self.token_pos = nn.Parameter(torch.randn(1, dim, *token_grid) * .02)
        self.intra = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(intra_layers)])
        self.token_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        b = self.c4(s3)
        # Convolutional patch embedding, 2x2x2 bottleneck voxels per token.
        # Right padding avoids dropping last voxels if spatial dimensions are odd.
        pad_d, pad_h, pad_w = (n % 2 for n in b.shape[2:])
        patch_map = self.patch_embed(F.pad(b, (0, pad_w, 0, pad_h, 0, pad_d)))
        spatial_shape = patch_map.shape[2:]
        position = F.interpolate(
            self.token_pos, size=spatial_shape, mode="trilinear", align_corners=False
        ).flatten(2).transpose(1, 2)
        tokens = patch_map.flatten(2).transpose(1, 2) + position
        for blk in self.intra:
            tokens = blk(tokens)
        tokens = self.token_norm(tokens)
        return (s1, s2, s3, b), tokens


class AnatomyInterModalTransformer(nn.Module):
    """mmFormer-style inter-modal attention augmented with anatomy visibility.

    Patch<->patch is permitted, anatomy<->patch only for visible views;
    anatomy<->anatomy is permitted. Missing views are masked as keys/values.
    Missing patch queries are zeroed after fusion.
    """
    def __init__(self, dim: int, heads: int, layers: int,
                 dropout: float = .1):
        super().__init__()
        self.register_buffer("visibility", torch.tensor(VISIBILITY, dtype=torch.bool))
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(layers)])
        self.end_norm = nn.LayerNorm(dim)
        self.num_heads = heads

    def _make_mask(self, p: int, device: torch.device) -> Tensor:
        n = 3*p+5
        mask = torch.ones(n, n, dtype=torch.bool, device=device)
        mask[:3*p, :3*p] = False
        mask[3*p:, 3*p:] = False
        for a in range(5):
            for v in range(3):
                if VISIBILITY[a][v]:
                    mask[3*p+a, v*p:(v+1)*p] = False
                    mask[v*p:(v+1)*p, 3*p+a] = False
        return mask

    def forward(self, tokens: Tensor, anatomy: Tensor, present: Tensor):
        b, v, p, dim = tokens.shape
        assert v == 3
        # For an anatomy query with zero visible modalities, allow its self-key.
        # For any missing patches, other tokens cannot read them.
        pad = torch.cat(((~present.bool()).repeat_interleave(p, dim=1),
                         torch.zeros(b, 5, dtype=torch.bool, device=tokens.device)), dim=1)
        seq = torch.cat((tokens.reshape(b, 3*p, dim), anatomy), dim=1)
        structure_mask = self._make_mask(p, tokens.device)
        for layer in self.layers:
            seq = layer(seq, attn_mask=structure_mask, key_padding_mask=pad)
        seq = self.end_norm(seq)
        patch = seq[:, :3*p].reshape(b, 3, p, dim)
        patch = patch * present[:, :, None, None].to(patch.dtype)
        return patch, seq[:, 3*p:]


class UpFuse(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.fuse = ConvBlock(cin+cskip, cout)

    def forward(self, x: Tensor, skip: Tensor):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat((x, skip), 1))


class ProgressiveDecoder(nn.Module):
    """Shared progressive upsampling decoder with anatomy FiLM conditioning."""
    # 增加 film_hidden_dim 参数，通常可设为与 dim 相同或更大的值
    def __init__(self, base: int, dim: int, film_hidden_dim: int = 128):
        super().__init__()
        self.d3 = UpFuse(dim, base*4, base*4)
        self.d2 = UpFuse(base*4, base*2, base*2)
        self.d1 = UpFuse(base*2, base, base)
        
        # 1. 使用 MLP (Linear -> GELU -> Linear) 替代单一 nn.Linear
        self.films = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, 2 * c)
            ) for c in (base*4, base*2, base)
        ])
        
        # 2. 初始化策略调整：仅将最后一层初始化为全零
        for film in self.films:
            # 保持前置隐层的默认非零初始化，确保梯度能穿透网络回传给解剖条件特征
            # 仅将输出层的权重和偏置设为 0，使初始状态下 gamma=0, beta=0
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

    @staticmethod
    def modulate(x: Tensor, film: nn.Module, cond: Tensor):
        gamma, beta = film(cond).chunk(2, dim=-1)
        return x * (1 + gamma[..., None, None, None]) + beta[..., None, None, None]

    def forward(self, skips: Sequence[Tensor], cond: Optional[Tensor] = None):
        s1, s2, s3, bottleneck = skips
        d3 = self.d3(bottleneck, s3)
        if cond is not None: d3 = self.modulate(d3, self.films[0], cond)
        d2 = self.d2(d3, s2)
        if cond is not None: d2 = self.modulate(d2, self.films[1], cond)
        d1 = self.d1(d2, s1)
        if cond is not None: d1 = self.modulate(d1, self.films[2], cond)
        return d1, d2, d3

class MMFormerAnatomy(nn.Module):
    """Three-view, anatomy-aware mmFormer adaptation.

    Args:
        token_grid: reference grid for positional embeddings ONLY;
          token count is determined by the stride-2 patch convolution output.
        dim: Transformer / bottleneck dimension (divisible by heads).
        base: CNN width. 16/64 are lightweight development defaults.
        predict_missing: unavailable views are zeroed in returned logits;
          model is designed to segment only observed views.
    """
    def __init__(self, in_channels: int = 1,
                 num_classes_by_source: Optional[Dict[str,int]] = None,
                 source_order: Sequence[str] = VIEWS,
                 base: int = 32, dim: int = 64, heads: int = 4,
                 token_grid: Tuple[int,int,int] = (2,4,4),
                 intra_layers: int = 1, inter_layers: int = 2,
                 predict_missing: bool = False):
        super().__init__()
        if tuple(source_order) != VIEWS:
            raise ValueError("source_order must be ('2ch','4ch','sa') for fixed anatomy visibility")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.source_order = list(source_order)
        self.num_classes_by_source = num_classes_by_source or {"2ch":3, "4ch":6, "sa":4}
        self.token_grid = token_grid
        self.p = None  # Dynamic: product of patch-convolution output dimensions
        self.base = base
        self.dim = dim
        self.predict_missing = predict_missing
        self.encoders = nn.ModuleDict({v: HybridViewEncoder(in_channels, base, dim, token_grid, intra_layers, heads)
                                        for v in VIEWS})
        self.modality_embedding = nn.Parameter(torch.randn(3,1,dim)*.02)
        self.anatomy_tokens = nn.Parameter(torch.randn(1,5,dim)*.02)
        self.anatomy_pos = nn.Parameter(torch.randn(1,5,dim)*.02)
        self.inter = AnatomyInterModalTransformer(dim, heads, inter_layers)
        self.cond_proj = nn.ModuleDict({v: nn.Sequential(nn.Linear(len(ids)*dim,dim), nn.LayerNorm(dim),nn.GELU())
                                        for v,ids in {"2ch":(0,1),"4ch":(0,1,2,3,4),"sa":(0,1,2)}.items()})
        self.anatomy_indices = {"2ch":(0,1), "4ch":(0,1,2,3,4), "sa":(0,1,2)}
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.decoder = ProgressiveDecoder(base, dim)
        self.heads = nn.ModuleDict({v: nn.Conv3d(base,self.num_classes_by_source[v],1) for v in VIEWS})
        self.deep_heads = nn.ModuleDict({v: nn.ModuleList([
            nn.Conv3d(base*2,self.num_classes_by_source[v],1),
            nn.Conv3d(base*4,self.num_classes_by_source[v],1)]) for v in VIEWS})
        # Encoder auxiliary decoder has SHARED weights, view-specific prediction heads.
        self.aux_decoder = ProgressiveDecoder(base, dim)
        self.aux_heads = nn.ModuleDict({v: nn.Conv3d(base,self.num_classes_by_source[v],1) for v in VIEWS})

    def forward(self, x: Tensor, modality_mask: Tensor, full_x: Optional[Tensor] = None) -> Dict:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("x must have shape [B,3,D,H,W]")
        b = x.shape[0]
        if modality_mask.shape != (b,3):
            raise ValueError(f"modality_mask must have shape [{b},3]")
        present = modality_mask.to(device=x.device, dtype=torch.bool)
        if not present.any(dim=1).all():
            raise ValueError("Each sample must contain at least one available view")
        # Physically zero missing inputs, even if caller passed nonzero content.
        x = x * present[:, :, None, None, None].to(x.dtype)
        skip_maps = {}
        view_tokens = []
        # Infer dimensions analytically from the three stride-2 ConvBlocks.
        # Formula for kernel=3, padding=1, stride=2: ceil(size / 2).
        D, H, W = x.shape[2:]
        def half_size(shape):
            return tuple((n + 1) // 2 for n in shape)
        shape1 = (D, H, W)
        shape2 = half_size(shape1)
        shape3 = half_size(shape2)
        shape4 = half_size(shape3)
        spatial_shapes = (shape1, shape2, shape3, shape4)
        feature_channels = (self.base, 2*self.base, 4*self.base, self.dim)
        patch_shape = tuple((n + 1) // 2 for n in shape4)
        p = patch_shape[0] * patch_shape[1] * patch_shape[2]
        for i, view in enumerate(VIEWS):
            # Sub-batch ONLY valid patients so unavailable views never influence norm statistics.
            indices = present[:, i].nonzero(as_tuple=True)[0]
            # Allocate zeros without requiring a dummy forward through absent encoder.
            # Avoid in-place scatter side effects by using index_copy.
            if indices.numel() > 0:
                src_skips, src_tokens = self.encoders[view](x.index_select(0, indices)[:, i:i+1])
                skips = tuple(torch.zeros((b, *f.shape[1:]), device=f.device, dtype=f.dtype)
                              .index_copy(0, indices, f) for f in src_skips)
                tok = torch.zeros((b,p,src_tokens.shape[-1]),device=x.device,dtype=src_tokens.dtype)
                tok = tok.index_copy(0,indices,src_tokens)
            else:
                # No dummy inference (and no duplicate normalization passes).
                skips = tuple(x.new_zeros((b, c, *shape))
                              for c, shape in zip(feature_channels, spatial_shapes))
                tok = x.new_zeros((b, p, self.anatomy_tokens.shape[-1]))
            skip_maps[view] = skips
            view_tokens.append(tok + self.modality_embedding[i] * present[:,i,None,None])
        raw_tokens = torch.stack(view_tokens, dim=1)
        anatomy_init = (self.anatomy_tokens+self.anatomy_pos).expand(b,-1,-1)
        fused_tokens, anatomy = self.inter(raw_tokens,anatomy_init,present)
        logits, aux_logits, deep_logits, pred_tokens = {},{},{},{}
        for i, view in enumerate(VIEWS):
            pred_tokens[view] = fused_tokens[:,i]
            indices = present[:, i].nonzero(as_tuple=True)[0]
            s1,s2,s3,bottleneck = skip_maps[view]
            out_size = x.shape[2:]
            nclasses = self.num_classes_by_source[view]
            if indices.numel() == 0 and not self.predict_missing:
                logits[view] = x.new_zeros((b,nclasses,*out_size))
                aux_logits[view] = x.new_zeros((b,nclasses,*out_size))
                deep_logits[view] = [x.new_zeros((b,nclasses,*out_size)) for _ in range(2)]
                continue
            # Decode only observed samples; avoids propagation of missing-view features.
            # If predict_missing=True, missing-view decoding is not implemented intentionally.
            if self.predict_missing:
                raise NotImplementedError("Missing-view synthesis requires a dedicated reconstruction decoder")
            sk = [f.index_select(0, indices) for f in (s1,s2,s3,bottleneck)]
            patch = fused_tokens.index_select(0, indices)[:,i].transpose(1,2)
            patch = patch.reshape(indices.numel(), self.dim, *patch_shape)
            context = F.interpolate(
                patch, size=sk[-1].shape[2:], mode="trilinear", align_corners=False
            )
            sk[-1] = sk[-1]+self.context_scale*context
            anat_sel = anatomy.index_select(0,indices)[:,self.anatomy_indices[view],:]
            cond = self.cond_proj[view](anat_sel.flatten(1))
            d1,d2,d3 = self.decoder(sk,cond)
            pred = self.heads[view](d1)
            pred = F.interpolate(pred,size=out_size,mode="trilinear",align_corners=False)
            out = x.new_zeros((b,nclasses,*out_size)).index_copy(0,indices,pred)
            logits[view]=out
            deep = []
            for deep_head,feat in zip(self.deep_heads[view],(d2,d3)):
                dp = F.interpolate(deep_head(feat),size=out_size,mode="trilinear",align_corners=False)
                deep.append(x.new_zeros((b,nclasses,*out_size)).index_copy(0,indices,dp))
            deep_logits[view] = deep
            # mmFormer encoder-side auxiliary regularizer, no anatomy condition.
            aux1,_,_ = self.aux_decoder([f.index_select(0,indices) for f in (s1,s2,s3,bottleneck)])
            auxp = F.interpolate(self.aux_heads[view](aux1),size=out_size,mode="trilinear",align_corners=False)
            aux_logits[view]=x.new_zeros((b,nclasses,*out_size)).index_copy(0,indices,auxp)
        return {"logits":logits,"aux_logits":aux_logits,"deep_logits":deep_logits,
                "anatomy_tokens":anatomy,"pred_tokens":pred_tokens,"valid_views":present,
                "gt_tokens":{}}  # explicit: no teacher is implemented here


def masked_segmentation_loss(result: Dict, targets: Dict[str,Tensor],
                             aux_weight: float = .2, deep_weight: float = .1) -> Tensor:
    """Cross entropy over observed views only. Targets: per view [B,D,H,W], integer labels."""
    present = result["valid_views"]
    terms = []
    for i, view in enumerate(VIEWS):
        valid = present[:,i]
        if not valid.any():
            continue
        target = targets[view][valid].long()
        loss = F.cross_entropy(result["logits"][view][valid],target)
        loss = loss + aux_weight * F.cross_entropy(result["aux_logits"][view][valid],target)
        for pred in result["deep_logits"][view]:
            loss = loss + deep_weight * F.cross_entropy(pred[valid],target)
        terms.append(loss)
    if not terms:
        raise ValueError("No valid supervised views")
    return torch.stack(terms).mean()


if __name__ == "__main__":
    torch.manual_seed(1)
    model = MMFormerAnatomy(base=8, dim=32, heads=4, token_grid=(2,2,2), intra_layers=1, inter_layers=1)
    x = torch.randn(2,3,16,32,32)
    mask = torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
    output = model(x,mask)
    for view, pred in output["logits"].items():
        print(view, tuple(pred.shape))
    targets = {v:torch.randint(0,c,(2,16,32,32)) for v,c in model.num_classes_by_source.items()}
    loss = masked_segmentation_loss(output,targets)
    loss.backward()
    print("forward/backward OK, loss=",round(loss.item(),4))
