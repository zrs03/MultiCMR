"""Unrestricted multi-view cardiac MRI segmenter.

Three view-specific CNN/Transformer encoders (2ch, 4ch, sa), two layers of
unrestricted cross-view attention among available views, and a progressive
segmentation decoder. Missing views are masked. No anatomical routing.

Input x: [B,3,D,H,W], modality_mask: [B,3].
Training: masked_segmentation_loss(output, targets).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from torch import Tensor, nn

import torch.nn.functional as F

VIEWS = ("2ch", "4ch", "sa")

ANATOMY = ("background", "LV_myo", "LV_cav", "RV_cav", "RA", "LA")

DEFAULT_LABEL_MAP = {"2ch": (0, 1, 2), "4ch": (0, 1, 2, 3, 4, 5), "sa": (0, 1, 2, 3)}

def norm(c: int) -> nn.Module:

    for g in (8, 4, 2, 1):

        if c % g == 0:

            return nn.GroupNorm(g, c)

    return nn.GroupNorm(1, c)

class ConvBlock(nn.Module):

    def __init__(self, cin: int, cout: int, stride: int = 1):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(cin, cout, 3, stride, 1, bias=False), norm(cout), nn.ReLU(inplace=True),

            nn.Conv3d(cout, cout, 3, 1, 1, bias=False), norm(cout),

        )

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

        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),

                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,

                key_padding_mask: Optional[Tensor] = None) -> Tensor:

        y = self.n1(x)

        y, _ = self.attn(y, y, y, attn_mask=attn_mask,

                         key_padding_mask=key_padding_mask, need_weights=False)

        x = x + y

        return x + self.mlp(self.n2(x))

class HybridViewEncoder(nn.Module):

    def __init__(self, in_channels: int, base: int, dim: int,

                 token_grid: Tuple[int, int, int], intra_layers: int, heads: int):

        super().__init__()

        self.c1 = ConvBlock(in_channels, base)

        self.c2 = ConvBlock(base, base * 2, 2)

        self.c3 = ConvBlock(base * 2, base * 4, 2)

        self.c4 = ConvBlock(base * 4, dim, 2)

        self.patch_embed = nn.Conv3d(dim, dim, kernel_size=1, stride=1)

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

                            mode="trilinear", align_corners=False)

        tokens = patch_map.flatten(2).transpose(1, 2) + pos.flatten(2).transpose(1, 2)

        for layer in self.intra:

            tokens = layer(tokens)

        return (s1, s2, s3, b), self.token_norm(tokens), token_shape

class TwoStageInterModalTransformer(nn.Module):
    """Two unrestricted cross-view attention layers.

    Present queries may attend to all present views. Missing-view queries only
    attend to keys in their own missing view, avoiding all-masked softmax rows;
    their outputs are set to zero. No anatomical routing is used.
    """
    def __init__(self, dim: int, heads: int, dropout: float = .1):
        super().__init__()
        self.first = TransformerBlock(dim, heads, dropout)
        self.second = TransformerBlock(dim, heads, dropout)
        self.end_norm = nn.LayerNorm(dim)
        self.heads = heads

    def forward(self, tokens: Tensor, present: Tensor) -> Tensor:
        batch, views, patches, dim = tokens.shape
        if views != 3 or present.shape != (batch, 3):
            raise ValueError("Expected tokens [B,3,P,C], present [B,3]")
        n = views * patches
        view_ids = torch.arange(3, device=tokens.device).repeat_interleave(patches)
        same_view = view_ids[:, None] == view_ids[None, :]
        missing = (~present.bool()).repeat_interleave(patches, dim=1)
        absent_keys = missing[:, None, :].expand(batch, n, n)
        absent_queries = missing[:, :, None].expand(batch, n, n)
        # Available queries cannot read missing keys. Missing queries can read
        # their own missing-view keys only, so softmax always has a valid key.
        block = (absent_keys & ~absent_queries) | (absent_queries & ~same_view.unsqueeze(0))
        mask = block.repeat_interleave(self.heads, dim=0)
        flat = tokens.reshape(batch, n, dim)
        flat = self.first(flat, attn_mask=mask)
        flat = self.second(flat, attn_mask=mask)
        result = self.end_norm(flat).reshape(batch, 3, patches, dim)
        return result * present[:, :, None, None].to(result.dtype)

class UpFuse(nn.Module):

    def __init__(self, cin: int, cskip: int, cout: int):

        super().__init__()

        self.fuse = ConvBlock(cin + cskip, cout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:

        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        return self.fuse(torch.cat((x, skip), dim=1))

class ProgressiveDecoder(nn.Module):

    def __init__(self, base: int, dim: int):

        super().__init__()

        self.d3 = UpFuse(dim, base * 4, base * 4)

        self.d2 = UpFuse(base * 4, base * 2, base * 2)

        self.d1 = UpFuse(base * 2, base, base)

    def forward(self, skips: Sequence[Tensor]):

        s1, s2, s3, bottleneck = skips

        d3 = self.d3(bottleneck, s3)

        d2 = self.d2(d3, s2)

        d1 = self.d1(d2, s1)

        return d1, d2, d3

class MMFormerAnatomy(nn.Module):

    def __init__(self, in_channels: int = 1,

                 num_classes_by_source: Optional[Dict[str, int]] = None,

                 source_order: Sequence[str] = VIEWS,

                 base: int = 32, dim: int = 128, heads: int = 4,

                 token_grid: Tuple[int, int, int] = (2, 4, 4),

                 intra_layers: int = 1, inter_layers: int = 2,

                 label_to_anatomy: Optional[Mapping[str, Sequence[int]]] = None,

                 predict_missing: bool = False):

        super().__init__()

        if tuple(source_order) != VIEWS:

            raise ValueError(f"source_order must be {VIEWS}")

        if dim % heads:

            raise ValueError("dim must be divisible by heads")

        if inter_layers != 2:

            raise ValueError("Exactly 2 inter-modal layers are required (restricted then unrestricted)")

        if predict_missing:

            raise ValueError("Missing-view synthesis is not supported; only observed views are segmented")

        self.num_classes_by_source = num_classes_by_source or {"2ch": 3, "4ch": 6, "sa": 4}

        mapping = dict(label_to_anatomy) if label_to_anatomy is not None else DEFAULT_LABEL_MAP

        for view in VIEWS:

            if len(mapping[view]) != self.num_classes_by_source[view]:

                raise ValueError(f"label_to_anatomy[{view}] length must match number of segmentation classes")

            if any(a < 0 or a >= len(ANATOMY) for a in mapping[view]):

                raise ValueError(f"Invalid anatomy IDs for {view}")

            self.register_buffer(f"label_map_{view}", torch.tensor(mapping[view], dtype=torch.long))

        self.base, self.dim = base, dim

        self.encoders = nn.ModuleDict({v: HybridViewEncoder(in_channels, base, dim,

                                       token_grid, intra_layers, heads) for v in VIEWS})

        self.modality_embedding = nn.Parameter(torch.randn(3, 1, dim) * .02)

        self.inter = TwoStageInterModalTransformer(dim, heads)

        self.context_scale = nn.Parameter(torch.tensor(.1))

        self.decoder = ProgressiveDecoder(base, dim)

        self.heads = nn.ModuleDict({v: nn.Conv3d(base, self.num_classes_by_source[v], 1) for v in VIEWS})

        self.deep_heads = nn.ModuleDict({v: nn.ModuleList([

            nn.Conv3d(base * 2, self.num_classes_by_source[v], 1),

            nn.Conv3d(base * 4, self.num_classes_by_source[v], 1)]) for v in VIEWS})

        self.aux_decoder = ProgressiveDecoder(base, dim)

        self.aux_heads = nn.ModuleDict({v: nn.Conv3d(base, self.num_classes_by_source[v], 1) for v in VIEWS})

    def forward(self, x: Tensor, modality_mask: Tensor, full_x: Optional[Tensor] = None) -> Dict:

        if x.ndim != 5 or x.shape[1] != 3:

            raise ValueError("x must have shape [B,3,D,H,W]")

        batch = x.shape[0]

        if modality_mask.shape != (batch, 3):

            raise ValueError("modality_mask must have shape [B,3]")

        present = modality_mask.to(device=x.device, dtype=torch.bool)

        if not present.any(dim=1).all():

            raise ValueError("Each sample must have at least one available view")

        x = x * present[:, :, None, None, None].to(x.dtype)

        D, H, W = x.shape[2:]

        def half(sh):

            return tuple((n + 1) // 2 for n in sh)

        spatial_shapes = [(D, H, W)]

        for _ in range(3):

            spatial_shapes.append(half(spatial_shapes[-1]))

        patch_shape = spatial_shapes[-1]

        patches = patch_shape[0] * patch_shape[1] * patch_shape[2]

        chans = (self.base, self.base*2, self.base*4, self.dim)

        skip_maps, view_tokens = {}, []

        for i, view in enumerate(VIEWS):

            indices = present[:, i].nonzero(as_tuple=True)[0]

            if indices.numel():

                sk, tok, tokshape = self.encoders[view](x.index_select(0, indices)[:, i:i+1])

                if tuple(tokshape) != tuple(patch_shape):

                    raise RuntimeError("Token grid mismatch")

                skips = tuple(f.new_zeros((batch, *f.shape[1:])).index_copy(0, indices, f) for f in sk)

                tokfull = tok.new_zeros((batch, patches, self.dim)).index_copy(0, indices, tok)

            else:

                skips = tuple(x.new_zeros((batch, c, *sh)) for c, sh in zip(chans, spatial_shapes))

                tokfull = x.new_zeros((batch, patches, self.dim))

            skip_maps[view] = skips

            view_tokens.append(tokfull + self.modality_embedding[i] * present[:, i, None, None])

        tokens = torch.stack(view_tokens, dim=1)

        fused = self.inter(tokens, present)

        logits, aux_logits, deep_logits, pred_tokens = {}, {}, {}, {}

        for i, view in enumerate(VIEWS):

            pred_tokens[view] = fused[:, i]

            indices = present[:, i].nonzero(as_tuple=True)[0]

            nclasses = self.num_classes_by_source[view]

            if not indices.numel():

                logits[view] = x.new_zeros((batch, nclasses, D, H, W))

                aux_logits[view] = x.new_zeros((batch, nclasses, D, H, W))

                deep_logits[view] = [x.new_zeros((batch, nclasses, D, H, W)) for _ in range(2)]

                continue

            sk = [f.index_select(0, indices) for f in skip_maps[view]]

            patch = fused.index_select(0, indices)[:, i].transpose(1, 2)

            patch = patch.reshape(indices.numel(), self.dim, *patch_shape)

            context = F.interpolate(patch, size=sk[-1].shape[2:], mode="trilinear", align_corners=False)

            sk[-1] = sk[-1] + self.context_scale * context

            d1, d2, d3 = self.decoder(sk)

            pred = F.interpolate(self.heads[view](d1), size=(D, H, W),

                                 mode="trilinear", align_corners=False)

            logits[view] = pred.new_zeros((batch, nclasses, D, H, W)).index_copy(0, indices, pred)

            deep = []

            for dh, feat in zip(self.deep_heads[view], (d2, d3)):

                dp = F.interpolate(dh(feat), size=(D, H, W), mode="trilinear", align_corners=False)

                deep.append(dp.new_zeros((batch, nclasses, D, H, W)).index_copy(0, indices, dp))

            deep_logits[view] = deep

            if self.training:

                aux1, _, _ = self.aux_decoder([f.index_select(0, indices) for f in skip_maps[view]])

                ap = F.interpolate(self.aux_heads[view](aux1), size=(D, H, W),

                                   mode="trilinear", align_corners=False)

                aux_logits[view] = ap.new_zeros((batch, nclasses, D, H, W)).index_copy(0, indices, ap)

        return dict(logits=logits, aux_logits=aux_logits, deep_logits=deep_logits,

                    pred_tokens=pred_tokens, valid_views=present,

                    patch_shape=patch_shape)

def masked_segmentation_loss(result: Dict, targets: Dict[str, Tensor],
                             aux_weight: float = .2, deep_weight: float = .1) -> Tensor:
    """Cross-entropy segmentation loss, averaged across available views.

    Includes training-time auxiliary and deep supervision terms when present.
    """
    present = result["valid_views"]
    terms = []
    for i, view in enumerate(VIEWS):
        valid = present[:, i]
        if not valid.any():
            continue
        target = targets[view][valid].long()
        loss = F.cross_entropy(result["logits"][view][valid], target)
        if view in result["aux_logits"]:
            loss = loss + aux_weight * F.cross_entropy(result["aux_logits"][view][valid], target)
        for dp in result["deep_logits"][view]:
            loss = loss + deep_weight * F.cross_entropy(dp[valid], target)
        terms.append(loss)
    if not terms:
        raise ValueError("No valid supervised views")
    return torch.stack(terms).mean()

if __name__ == "__main__":

    torch.manual_seed(1)

    model = MMFormerAnatomy(base=8, dim=32, heads=4, token_grid=(2, 2, 2),

                             intra_layers=1, inter_layers=2)

    x = torch.randn(2, 3, 16, 32, 32)

    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)

    output = model(x, mask)

    for view, pred in output["logits"].items():

        print(view, tuple(pred.shape))

    targets = {v: torch.randint(0, c, (2, 16, 32, 32))

               for v, c in model.num_classes_by_source.items()}

    loss = masked_segmentation_loss(output, targets)

    loss.backward()

    assert torch.isfinite(loss)

    model.eval()

    with torch.no_grad():

        output = model(x, mask)

    print("forward/backward/eval OK; loss=", round(loss.item(), 4))
