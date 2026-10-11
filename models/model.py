"""Multi-view cardiac MRI segmentation with learnable fusion and anatomy tokens.
M2FTrans-inspired: P learnable fusion tokens for P spatial feature patches.
Additional: 5 learnable anatomy tokens (LV_myo, LV_cav, RV_cav, RA, LA).
No hard anatomical routing. Missing views excluded as attention keys.
Anatomy tokens are injected through class-aware decoder cross-attention.
This is a research adaptation, not an exact reimplementation of M2FTrans.
"""
from __future__ import annotations
from math import prod

from typing import Dict, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
VIEWS = ("2ch", "4ch", "sa")
ANATOMY = ("background", "LV_myo", "LV_cav", "RV_cav", "RA", "LA")
DEFAULT_LABEL_MAP = {"2ch": (0, 1, 2), "4ch": (0, 1, 2, 3, 4, 5), "sa": (0, 1, 2, 3)}


def norm(c: int) -> nn.Module:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, spatial_dims: int = 3):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv = nn.Sequential(
            conv(cin, cout, 3, stride, 1, bias=False), norm(cout), nn.ReLU(inplace=True),
            conv(cout, cout, 3, 1, 1, bias=False), norm(cout),
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else conv(cin, cout, 1, stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1, query_chunk_size: int = 256):
        super().__init__()
        self.query_chunk_size = query_chunk_size
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        y = self.n1(x)
        if y.shape[1] <= self.query_chunk_size or attn_mask is not None:
            y, _ = self.attn(y, y, y, attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask, need_weights=False)
        else:
            # Exact global attention with bounded query chunks. Each chunk still
            # attends ALL keys; this never pools/drops patches or separates time.
            batch, length, dim = y.shape
            heads = self.attn.num_heads
            q, k, v = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, -1)
            q, k, v = (a.reshape(batch, length, heads, dim // heads).transpose(1, 2)
                       for a in (q, k, v))
            allowed = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]

            def attend(query, key, value):
                return F.scaled_dot_product_attention(
                    query, key, value, attn_mask=allowed,
                    dropout_p=self.attn.dropout if self.training else 0.)

            chunks = []
            for query in q.split(self.query_chunk_size, dim=2):
                if self.training and torch.is_grad_enabled():
                    chunk = checkpoint(attend, query, k, v, use_reentrant=False)
                else:
                    chunk = attend(query, k, v)
                chunks.append(chunk)
            y = torch.cat(chunks, dim=2).transpose(1, 2).reshape(batch, length, dim)
            y = self.attn.out_proj(y)
        x = x + y
        return x + self.mlp(self.n2(x))


class HybridViewEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int, dim: int,
                 token_grid: Tuple[int, ...], intra_layers: int, heads: int,
                 spatial_dims: int = 3):
        super().__init__()
        self.spatial_dims = spatial_dims
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.c1 = ConvBlock(in_channels, base, spatial_dims=spatial_dims)
        self.c2 = ConvBlock(base, base * 2, 2, spatial_dims)
        self.c3 = ConvBlock(base * 2, base * 4, 2, spatial_dims)
        self.c4 = ConvBlock(base * 4, dim, 2, spatial_dims)
        self.patch_embed = conv(dim, dim, kernel_size=1, stride=1)
        self.token_pos = nn.Parameter(torch.randn(1, dim, *token_grid) * .02)
        self.intra = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(intra_layers)])
        self.token_norm = nn.LayerNorm(dim)
    def forward(self, x: Tensor):
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        b = self.c4(s3)
        patch_map = self.patch_embed(b)
        token_shape = patch_map.shape[2:]
        pos = F.interpolate(self.token_pos, size=token_shape,
                            mode="bilinear" if self.spatial_dims == 2 else "trilinear", align_corners=False)
        tokens = patch_map.flatten(2).transpose(1, 2) + pos.flatten(2).transpose(1, 2)
        for layer in self.intra:
            tokens = layer(tokens)
        return (s1, s2, s3, b), self.token_norm(tokens), token_shape


class CineViewEncoder(HybridViewEncoder):
    """Shared 2D encoder for every frame, followed by temporal patch concatenation."""
    def __init__(self, in_channels, base, dim, token_grid, intra_layers, heads,
                 temporal_size=64):
        super().__init__(in_channels, base, dim, token_grid[-2:], intra_layers, heads,
                         spatial_dims=2)
        self.temporal_pos = nn.Parameter(torch.randn(1, dim, temporal_size) * .02)

    def forward(self, x: Tensor):
        batch, channels, time, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        frame_skips, tokens, spatial_shape = super().forward(frames)
        # Keep every frame's spatial patches in t-major order: [t0 patches, t1 patches, ...].
        tokens = tokens.reshape(batch, time, -1, tokens.shape[-1])
        temporal_pos = F.interpolate(self.temporal_pos, size=time, mode='linear', align_corners=False)
        tokens = tokens + temporal_pos.transpose(1, 2)[:, :, None, :]
        tokens = tokens.flatten(1, 2)
        skips = tuple(f.reshape(batch, time, *f.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
                      for f in frame_skips)
        return skips, tokens, (time, *spatial_shape)


class UpFuse(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int, spatial_dims: int = 3):
        super().__init__()
        self.mode = "bilinear" if spatial_dims == 2 else "trilinear"
        self.fuse = ConvBlock(cin + cskip, cout, spatial_dims=spatial_dims)
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode=self.mode, align_corners=False)
        return self.fuse(torch.cat((x, skip), dim=1))


class ProgressiveDecoder(nn.Module):
    def __init__(self, base: int, dim: int, spatial_dims: int = 3):
        super().__init__()
        self.d3 = UpFuse(dim, base * 4, base * 4, spatial_dims)
        self.d2 = UpFuse(base * 4, base * 2, base * 2, spatial_dims)
        self.d1 = UpFuse(base * 2, base, base, spatial_dims)
    def forward(self, skips: Sequence[Tensor]):
        s1, s2, s3, bottleneck = skips
        d3 = self.d3(bottleneck, s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)
        return d1, d2, d3


class FusionAnatomyTransformer(nn.Module):
    """Joint self attention over [three-view patches, P fusion tokens, 5 anatomy tokens]."""
    def __init__(self, dim: int, heads: int, grid: Tuple[int, int, int], layers: int = 2,
                 dropout: float = .1):
        super().__init__()
        self.fusion_grid = nn.Parameter(torch.randn(1, dim, *grid) * .02)
        self.anatomy_tokens = nn.Parameter(torch.randn(1, 5, dim) * .02)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
    def forward(self, tokens: Mapping[str, Tensor], present: Tensor,
                patch_shape: Tuple[int, int, int]):
        """Keep native 2D/3D patches; fusion tokens use the SAX bottleneck grid."""
        b = tokens['sa'].shape[0]
        if present.shape != (b, 3):
            raise ValueError('Expected present [B,3]')
        lengths = [tokens[v].shape[1] for v in VIEWS]
        p = prod(patch_shape)
        fusion = F.interpolate(self.fusion_grid, size=patch_shape, mode='trilinear',
                               align_corners=False).flatten(2).transpose(1, 2).expand(b, -1, -1)
        anatomy = self.anatomy_tokens.expand(b, -1, -1)
        x = torch.cat([*(tokens[v] for v in VIEWS), fusion, anatomy], dim=1)
        missing = torch.cat([
            *(~present[:, i:i+1].bool().expand(-1, n) for i, n in enumerate(lengths)),
            torch.zeros(b, p + 5, device=x.device, dtype=torch.bool),
        ], dim=1)
        # Missing tokens never act as keys/values; their query outputs are discarded.
        # Fusion/anatomy keys remain available, so no softmax row is fully masked.
        for layer in self.layers:
            x = layer(x, key_padding_mask=missing)
        chunks = self.norm(x).split([*lengths, p, 5], dim=1)
        views_out = {v: chunks[i] * present[:, i, None, None].to(x.dtype)
                     for i, v in enumerate(VIEWS)}
        return views_out, chunks[-2], chunks[-1]


class AnatomyDecoderInjection(nn.Module):
    """Voxel features query anatomy tokens; residual anatomy context is injected."""
    def __init__(self, channels: int, dim: int, spatial_dims: int = 3):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.query = conv(channels, dim, kernel_size=1)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, channels)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.scale = dim ** -0.5
    def forward(self, feat: Tensor, anatomy: Tensor, class_available: Tensor):
        b, channels = feat.shape[:2]
        q = self.query(feat).flatten(2).transpose(1, 2)  # [B, voxels, dim]
        k = self.key(anatomy)                           # [B, 5, dim]
        v = self.value(anatomy)                         # [B, 5, channels]
        scores = (q @ k.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~class_available[None, None, :],
                                    torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(b, channels, *feat.shape[2:])
        return feat + torch.tanh(self.gate) * context


class AnatomyAwareProgressiveDecoder(nn.Module):
    def __init__(self, base: int, dim: int, spatial_dims: int = 3):
        super().__init__()
        self.d3 = UpFuse(dim, base * 4, base * 4, spatial_dims)
        self.d2 = UpFuse(base * 4, base * 2, base * 2, spatial_dims)
        self.d1 = UpFuse(base * 2, base, base, spatial_dims)
        self.a3 = AnatomyDecoderInjection(base * 4, dim, spatial_dims)
        self.a2 = AnatomyDecoderInjection(base * 2, dim, spatial_dims)
        self.a1 = AnatomyDecoderInjection(base, dim, spatial_dims)
    def forward(self, skips: Sequence[Tensor], anatomy: Tensor, availability: Tensor):
        s1, s2, s3, bottleneck = skips
        d3 = self.a3(self.d3(bottleneck, s3), anatomy, availability)
        d2 = self.a2(self.d2(d3, s2), anatomy, availability)
        d1 = self.a1(self.d1(d2, s1), anatomy, availability)
        return d1, d2, d3


class MMFormerAnatomy(nn.Module):
    """M2FTrans-inspired learnable fusion tokens + 5 anatomy tokens for 3 MRI views."""
    def __init__(self, in_channels: int = 1,
                 num_classes_by_source: Optional[Dict[str, int]] = None,
                 source_order: Sequence[str] = VIEWS,
                 base: int = 16, dim: int = 128, heads: int = 4,
                 token_grid: Tuple[int, int, int] = (2, 4, 4),
                 intra_layers: int = 1, inter_layers: int = 2,
                 label_to_anatomy: Optional[Mapping[str, Sequence[int]]] = None,
                 predict_missing: bool = False, temporal_size: int = 64):
        super().__init__()
        if tuple(source_order) != VIEWS:
            raise ValueError(f'source_order must be {VIEWS}')
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        if inter_layers < 1:
            raise ValueError('inter_layers must be positive')
        if predict_missing:
            raise ValueError('Missing-view synthesis is not supported')
        self.num_classes_by_source = num_classes_by_source or {'2ch': 3, '4ch': 6, 'sa': 4}
        mapping = dict(label_to_anatomy) if label_to_anatomy is not None else DEFAULT_LABEL_MAP
        for view in VIEWS:
            if len(mapping[view]) != self.num_classes_by_source[view]:
                raise ValueError(f'label mapping length mismatch for {view}')
            if any(a < 0 or a >= len(ANATOMY) for a in mapping[view]):
                raise ValueError(f'invalid anatomy IDs for {view}')
            valid = torch.tensor([a in mapping[view] for a in range(1, 6)], dtype=torch.bool)
            if not valid.any():
                raise ValueError(f'{view} must contain at least one nonbackground anatomy')
            self.register_buffer(f'anatomy_available_{view}', valid)
            self.register_buffer(f'label_map_{view}', torch.tensor(mapping[view], dtype=torch.long))
        self.base, self.dim = base, dim
        self.in_channels = in_channels
        self.encoders = nn.ModuleDict({
            v: (HybridViewEncoder(in_channels, base, dim, token_grid, intra_layers, heads)
                if v == 'sa' else CineViewEncoder(in_channels, base, dim, token_grid,
                                                  intra_layers, heads, temporal_size))
            for v in VIEWS})
        self.modality_embedding = nn.Parameter(torch.randn(3, 1, dim) * .02)
        self.inter = FusionAnatomyTransformer(dim, heads, token_grid, inter_layers)
        self.context_scale = nn.Parameter(torch.tensor(.1))
        self.fusion_scale = nn.Parameter(torch.tensor(.1))
        self.decoder = AnatomyAwareProgressiveDecoder(base, dim)
        self.decoder_2d = AnatomyAwareProgressiveDecoder(base, dim, spatial_dims=2)
        self.heads = nn.ModuleDict({v: (nn.Conv3d if v == "sa" else nn.Conv2d)(base, self.num_classes_by_source[v], 1)
                                    for v in VIEWS})
        self.deep_heads = nn.ModuleDict({v: nn.ModuleList([
            (nn.Conv3d if v == "sa" else nn.Conv2d)(base * 2, self.num_classes_by_source[v], 1),
            (nn.Conv3d if v == "sa" else nn.Conv2d)(base * 4, self.num_classes_by_source[v], 1)]) for v in VIEWS})
        self.aux_decoder = ProgressiveDecoder(base, dim)
        self.aux_decoder_2d = ProgressiveDecoder(base, dim, spatial_dims=2)
        self.aux_heads = nn.ModuleDict({v: (nn.Conv3d if v == "sa" else nn.Conv2d)(base, self.num_classes_by_source[v], 1)
                                        for v in VIEWS})
    def forward(self, x: Mapping[str, Tensor], modality_mask: Tensor) -> Dict:
        """2ch/4ch: [B,C,T,H,W] (full cine); sa: [B,C,D,H,W]."""
        if not isinstance(x, Mapping) or set(x) != set(VIEWS):
            raise ValueError('x must be a mapping with 2ch, 4ch and sa tensors')
        reference = x['sa']
        batch = reference.shape[0]
        for view in VIEWS:
            if x[view].ndim != 5 or x[view].shape[:2] != (batch, self.in_channels):
                raise ValueError(f'{view} expects [B,C,' + ('D,H,W]' if view == 'sa' else 'T,H,W]'))
            if x[view].device != reference.device:
                raise ValueError('All views must be on the same device')
        if modality_mask.shape != (batch, 3):
            raise ValueError('modality_mask must have shape [B,3]')
        present = modality_mask.to(device=reference.device, dtype=torch.bool)
        if not present.any(dim=1).all():
            raise ValueError('Each sample must have at least one available view')
        skip_maps, view_tokens, patch_shapes = {}, {}, {}
        chans = (self.base, self.base * 2, self.base * 4, self.dim)
        for i, view in enumerate(VIEWS):
            image = x[view]
            shapes = [tuple(image.shape[2:])]
            for _ in range(3):
                previous = shapes[-1]
                shapes.append(tuple((n + 1) // 2 for n in previous) if view == 'sa' else
                              (previous[0], *((n + 1) // 2 for n in previous[1:])))
            patch_shapes[view] = shapes[-1]
            patches = prod(shapes[-1])
            indices = present[:, i].nonzero(as_tuple=True)[0]
            if indices.numel():
                sk, tok, tokshape = self.encoders[view](image.index_select(0, indices))
                if tuple(tokshape) != shapes[-1]:
                    raise RuntimeError('Token grid mismatch')
                skips = tuple(f.new_zeros((batch, *f.shape[1:])).index_copy(0, indices, f)
                              for f in sk)
                tokfull = tok.new_zeros((batch, patches, self.dim)).index_copy(0, indices, tok)
            else:
                skips = tuple(image.new_zeros((batch, c, *sh)) for c, sh in zip(chans, shapes))
                tokfull = image.new_zeros((batch, patches, self.dim))
            skip_maps[view] = skips
            view_tokens[view] = tokfull + self.modality_embedding[i] * present[:, i, None, None]
        fused, fusion_tokens, anatomy_tokens = self.inter(view_tokens, present, patch_shapes['sa'])
        logits, aux_logits, deep_logits = {}, {}, {}
        shared_context = fusion_tokens.transpose(1, 2).reshape(batch, self.dim, *patch_shapes['sa'])
        for i, view in enumerate(VIEWS):
            indices = present[:, i].nonzero(as_tuple=True)[0]
            nclasses = self.num_classes_by_source[view]
            output_shape = (batch, nclasses, *x[view].shape[2:])
            if not indices.numel():
                logits[view] = reference.new_zeros(output_shape)
                if self.training:
                    aux_logits[view] = reference.new_zeros(output_shape)
                deep_logits[view] = [reference.new_zeros(output_shape) for _ in range(2)]
                continue
            is_2d = view != 'sa'
            mode = 'bilinear' if is_2d else 'trilinear'
            decoder = self.decoder_2d if is_2d else self.decoder
            aux_decoder = self.aux_decoder_2d if is_2d else self.aux_decoder
            sk = [f.index_select(0, indices) for f in skip_maps[view]]
            patch = fused[view].index_select(0, indices).transpose(1, 2)
            patch = patch.reshape(indices.numel(), self.dim, *patch_shapes[view])
            context = shared_context.index_select(0, indices)
            if is_2d:
                # SAX depth is not LAX time. Broadcast the depth-averaged fusion
                # context to every frame; the view tokens retain temporal context.
                context = F.interpolate(context.mean(dim=2), size=patch_shapes[view][-2:],
                                        mode='bilinear', align_corners=False).unsqueeze(2)
            else:
                context = F.interpolate(context, size=patch_shapes[view], mode=mode, align_corners=False)
            sk[-1] = sk[-1] + self.context_scale * patch + self.fusion_scale * context
            anatomy = anatomy_tokens.index_select(0, indices)
            time = x[view].shape[2]

            def frame_batch(feature):
                if not is_2d:
                    return feature
                return feature.permute(0, 2, 1, 3, 4).reshape(-1, feature.shape[1], *feature.shape[-2:])

            if is_2d:
                anatomy = anatomy.repeat_interleave(time, dim=0)
            d1, d2, d3 = decoder([frame_batch(f) for f in sk], anatomy,
                                  getattr(self, f'anatomy_available_{view}'))

            def restore(pred):
                size = output_shape[-2:] if is_2d else output_shape[2:]
                pred = F.interpolate(pred, size=size, mode=mode, align_corners=False)
                if is_2d:
                    pred = pred.reshape(indices.numel(), time, nclasses, *size).permute(0, 2, 1, 3, 4)
                return pred.new_zeros(output_shape).index_copy(0, indices, pred)

            logits[view] = restore(self.heads[view](d1))
            deep_logits[view] = [restore(head(feat))
                                 for head, feat in zip(self.deep_heads[view], (d2, d3))]
            if self.training:
                aux1, _, _ = aux_decoder([frame_batch(f.index_select(0, indices)) for f in skip_maps[view]])
                aux_logits[view] = restore(self.aux_heads[view](aux1))
        return dict(logits=logits, aux_logits=aux_logits, deep_logits=deep_logits,
                    pred_tokens=fused, fusion_tokens=fusion_tokens,
                    anatomy_tokens=anatomy_tokens, valid_views=present, patch_shapes=patch_shapes)


def masked_segmentation_loss(result: Dict, targets: Dict[str, Tensor],
                             aux_weight: float = .2, deep_weight: float = .1) -> Tensor:
    """Per-view segmentation CE, optional auxiliary and deep supervision."""
    terms = []
    for i, view in enumerate(VIEWS):
        valid = result['valid_views'][:, i]
        if not valid.any():
            continue
        target = targets[view][valid].long()
        loss = F.cross_entropy(result['logits'][view][valid], target)
        if view in result['aux_logits']:
            loss = loss + aux_weight * F.cross_entropy(result['aux_logits'][view][valid], target)
        for pred in result['deep_logits'][view]:
            loss = loss + deep_weight * F.cross_entropy(pred[valid], target)
        terms.append(loss)
    if not terms:
        raise ValueError('No valid supervised views')
    return torch.stack(terms).mean()

if __name__ == '__main__':
    torch.manual_seed(1)
    model = MMFormerAnatomy(base=8, dim=32, heads=4, token_grid=(2, 2, 2))
    x = {'2ch': torch.randn(2, 1, 3, 32, 32), '4ch': torch.randn(2, 1, 3, 32, 32),
         'sa': torch.randn(2, 1, 16, 32, 32)}
    present = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    result = model(x, present)
    print('Fusion:', tuple(result['fusion_tokens'].shape))
    print('Anatomy:', tuple(result['anatomy_tokens'].shape))
    for k, v in result['logits'].items():
        print(k, tuple(v.shape))
    targets = {v: torch.randint(0, c, (2, *x[v].shape[2:]))
               for v, c in model.num_classes_by_source.items()}
    loss = masked_segmentation_loss(result, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.inter.fusion_grid.grad is not None
    assert model.inter.anatomy_tokens.grad is not None
    model.eval()
    with torch.no_grad():
        result = model(x, present)
    print('forward/backward/eval OK; loss =', round(loss.item(), 4))
