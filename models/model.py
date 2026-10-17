"""Framewise LAX and 3D SAX with sequential anatomy cross-attention.
Five anatomy tokens condition per-view, per-scale decoder FiLM.
Missing views are excluded from attention keys in both stages.
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
        if y.shape[1] <= self.query_chunk_size:
            y, _ = self.attn(y, y, y, attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask, need_weights=False)
        else:
            # Exact attention with bounded query chunks. Each chunk sees all keys
            # allowed by the routing and missing-view masks; no patch pooling.
            batch, length, dim = y.shape
            heads = self.attn.num_heads
            q, k, v = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, -1)
            q, k, v = (a.reshape(batch, length, heads, dim // heads).transpose(1, 2)
                       for a in (q, k, v))
            if attn_mask is not None and (attn_mask.dtype != torch.bool or attn_mask.ndim != 2):
                raise ValueError('Chunked attention expects a 2D boolean blocked-edge mask')

            def attend(query, key, value, allowed):
                return F.scaled_dot_product_attention(
                    query, key, value, attn_mask=allowed,
                    dropout_p=self.attn.dropout if self.training else 0.)

            chunks = []
            for start in range(0, length, self.query_chunk_size):
                query = q[:, :, start:start + self.query_chunk_size]
                allowed = None
                if attn_mask is not None:
                    allowed = ~attn_mask[None, None, start:start + self.query_chunk_size, :]
                if key_padding_mask is not None:
                    keys_allowed = ~key_padding_mask[:, None, None, :]
                    allowed = keys_allowed if allowed is None else allowed & keys_allowed
                if self.training and torch.is_grad_enabled():
                    chunk = checkpoint(attend, query, k, v, allowed, use_reentrant=False)
                else:
                    chunk = attend(query, k, v, allowed)
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
        self.patch_embed = conv(dim, dim, kernel_size=2, stride=2)
        self.token_pos = nn.Parameter(torch.randn(1, dim, *token_grid) * .02)
        self.intra = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(intra_layers)])
        self.token_norm = nn.LayerNorm(dim)
    def forward(self, x: Tensor):
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        b = self.c4(s3)
        # Pad only the high end of odd spatial axes; retain all border features.
        padding = tuple(value for size in reversed(b.shape[2:]) for value in (0, size % 2))
        patch_map = self.patch_embed(F.pad(b, padding))
        token_shape = patch_map.shape[2:]
        pos = F.interpolate(self.token_pos, size=token_shape,
                            mode="bilinear" if self.spatial_dims == 2 else "trilinear", align_corners=False)
        tokens = patch_map.flatten(2).transpose(1, 2) + pos.flatten(2).transpose(1, 2)
        for layer in self.intra:
            tokens = layer(tokens)
        return (s1, s2, s3, b), self.token_norm(tokens), token_shape


class CineViewEncoder(HybridViewEncoder):
    """Shared framewise 2D encoder; time-major tokens and center-frame skips.

    B*T frames run in one batch. Intra attention is spatial per frame; anatomy cross-attention
    subsequently sees all T*H'*W' tokens with time positions.
    """
    def __init__(self, in_channels, base, dim, token_grid, intra_layers, heads,
                 temporal_size=5):
        if not isinstance(temporal_size, int) or (temporal_size < 1 or temporal_size % 2 == 0):
            raise ValueError('temporal_size must be a positive odd integer')
        super().__init__(in_channels, base, dim, token_grid[-2:],
                         intra_layers, heads, spatial_dims=2)
        self.in_channels = in_channels
        self.temporal_size = temporal_size
        self.temporal_pos = nn.Parameter(torch.randn(1, temporal_size, 1, dim) * .02)

    def forward(self, x: Tensor):
        if x.ndim == 4 and x.shape[1] == self.in_channels * self.temporal_size:
            x = x.reshape(x.shape[0], self.in_channels, self.temporal_size, *x.shape[-2:])
        if x.ndim == 5:
            if x.shape[1:3] != (self.in_channels, self.temporal_size):
                raise ValueError(f'Cine expects C={self.in_channels}, T={self.temporal_size}; '
                                 f'got {tuple(x.shape)}')
        else:
            raise ValueError(f'Cine expects [B,{self.in_channels * self.temporal_size},H,W] '
                             'or [B,C,T,H,W]')
        batch, channels, time, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        skips, tokens, spatial_shape = super().forward(frames)
        center_skips = tuple(s.reshape(batch, time, *s.shape[1:])[:, time // 2] for s in skips)
        tokens = tokens.reshape(batch, time, -1, tokens.shape[-1]) + self.temporal_pos
        return center_skips, tokens.flatten(1, 2), (time, *spatial_shape)


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
        return d1


class CrossAttentionBlock(nn.Module):
    """Rectangular cross-attention with unchanged queries when no evidence exists."""
    def __init__(self, dim, heads, dropout=.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, query, context, allowed):
        # allowed: [B, queries, keys]. Missing views and anatomy routing both apply.
        has_evidence = allowed.any(-1, keepdim=True)
        context = self.context_norm(context)
        # A zero fallback prevents all-masked softmax; its output is discarded.
        context = torch.cat((context, context.new_zeros(context.shape[0], 1, context.shape[-1])), 1)
        allowed = torch.cat((allowed, ~has_evidence), -1)
        blocked = (~allowed)[:, None].expand(-1, self.attn.num_heads, -1, -1)
        blocked = blocked.reshape(-1, query.shape[1], context.shape[1])
        update, _ = self.attn(self.query_norm(query), context, context,
                              attn_mask=blocked, need_weights=False)
        output = query + update
        output = output + self.mlp(self.output_norm(output))
        return torch.where(has_evidence, output, query)


class AnatomyCrossStage(nn.Module):
    """Anatomy gathers view evidence, then patches read the updated anatomy."""
    def __init__(self, dim, heads, dropout=.1):
        super().__init__()
        self.anatomy_reads = CrossAttentionBlock(dim, heads, dropout)
        self.patch_reads = CrossAttentionBlock(dim, heads, dropout)

    def forward(self, patches, anatomy, allowed):
        anatomy = self.anatomy_reads(anatomy, patches, allowed.transpose(1, 2))
        patches = self.patch_reads(patches, anatomy, allowed)
        return patches, anatomy


class TwoStageAnatomyTransformer(nn.Module):
    """LAX/anatomy cross-attention followed by SAX/anatomy cross-attention."""
    def __init__(self, dim: int, heads: int, view_anatomy: Tensor, layers: int = 2,
                 dropout: float = .1):
        super().__init__()
        if layers != 2:
            raise ValueError('inter_layers must be 2: one LAX stage, then one SAX stage')
        self.register_buffer('view_anatomy', view_anatomy.to(dtype=torch.bool).clone())
        self.anatomy_tokens = nn.Parameter(torch.randn(1, 5, dim) * .02)
        self.lax_stage = AnatomyCrossStage(dim, heads, dropout)
        self.sax_stage = AnatomyCrossStage(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Mapping[str, Tensor], present: Tensor):
        batch = tokens['sa'].shape[0]
        if present.shape != (batch, 3) or not present.bool().any(dim=1).all():
            raise ValueError('Expected nonempty present mask [B,3]')
        present = present.bool()
        patches = [tokens[v].masked_fill(~present[:, i, None, None], 0)
                   for i, v in enumerate(VIEWS)]
        edges = [self.view_anatomy[i][None, None, :].expand(batch, p.shape[1], -1)
                 & present[:, i, None, None] for i, p in enumerate(patches)]
        anatomy = self.anatomy_tokens.expand(batch, -1, -1)
        lax, anatomy = self.lax_stage(torch.cat(patches[:2], 1), anatomy, torch.cat(edges[:2], 1))
        sax, anatomy = self.sax_stage(patches[2], anatomy, edges[2])
        updated = (*lax.split([p.shape[1] for p in patches[:2]], dim=1), sax)
        views_out = {v: self.norm(updated[i]).masked_fill(~present[:, i, None, None], 0)
                     for i, v in enumerate(VIEWS)}
        return views_out, self.norm(anatomy)


class AnatomyFiLM(nn.Module):
    """Selected anatomy tokens generate spatially shared channel modulation."""
    def __init__(self, channels: int, dim: int, num_tokens: int):
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.token_norm = nn.LayerNorm(dim)
        self.generator = nn.Sequential(
            nn.Linear(num_tokens * dim, dim), nn.SiLU(), nn.Linear(dim, 2 * channels))
        # Identity at initialization: F' = (1 + delta_gamma) * F + beta.
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, feat: Tensor, anatomy: Tensor, class_available: Tensor):
        selected = anatomy[:, class_available.bool(), :]
        if selected.shape[1] != self.num_tokens:
            raise ValueError('Anatomy availability does not match FiLM token count')
        condition = self.token_norm(selected).flatten(1)
        delta_gamma, beta = self.generator(condition).chunk(2, dim=1)
        shape = (feat.shape[0], self.channels) + (1,) * (feat.ndim - 2)
        return feat * (1 + delta_gamma.reshape(shape)) + beta.reshape(shape)


class ViewSpatialAdapter(nn.Module):
    """Identity-initialized 2D residual adapter with depthwise/pointwise convolutions."""
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = norm(channels)
        self.activation = nn.SiLU()
        self.pointwise = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, feature: Tensor):
        residual = self.pointwise(self.activation(self.norm(self.depthwise(feature))))
        return feature + residual


class AnatomyAwareProgressiveDecoder(nn.Module):
    def __init__(self, base: int, dim: int, view_token_counts: Mapping[str, int],
                 spatial_dims: int = 3):
        super().__init__()
        self.d3 = UpFuse(dim, base * 4, base * 4, spatial_dims)
        self.d2 = UpFuse(base * 4, base * 2, base * 2, spatial_dims)
        self.d1 = UpFuse(base * 2, base, base, spatial_dims)
        # Convolutions can be shared across views, but FiLM generators are not.
        self.a3 = nn.ModuleDict({v: AnatomyFiLM(base * 4, dim, n) for v, n in view_token_counts.items()})
        self.a2 = nn.ModuleDict({v: AnatomyFiLM(base * 2, dim, n) for v, n in view_token_counts.items()})
        self.a1 = nn.ModuleDict({v: AnatomyFiLM(base, dim, n) for v, n in view_token_counts.items()})
        # LAX shares the upsampling convolutions; spatial adapters are independent.
        self.spatial_adapters = nn.ModuleDict({
            v: nn.ModuleList([ViewSpatialAdapter(c) for c in (base * 4, base * 2, base)])
            for v in view_token_counts if spatial_dims == 2
        })

    def forward(self, skips: Sequence[Tensor], anatomy: Tensor, availability: Tensor, view: str):
        s1, s2, s3, bottleneck = skips
        d3 = self.a3[view](self.d3(bottleneck, s3), anatomy, availability)
        if view in self.spatial_adapters:
            d3 = self.spatial_adapters[view][0](d3)
        d2 = self.a2[view](self.d2(d3, s2), anatomy, availability)
        if view in self.spatial_adapters:
            d2 = self.spatial_adapters[view][1](d2)
        d1 = self.a1[view](self.d1(d2, s1), anatomy, availability)
        if view in self.spatial_adapters:
            d1 = self.spatial_adapters[view][2](d1)
        return d1


class MMFormerAnatomy(nn.Module):
    """LAX/anatomy then SAX/anatomy cross-attention and decoder FiLM."""
    def __init__(self, in_channels: int = 1,
                 num_classes_by_source: Optional[Dict[str, int]] = None,
                 source_order: Sequence[str] = VIEWS,
                 base: int = 16, dim: int = 128, heads: int = 4,
                 token_grid: Tuple[int, int, int] = (2, 4, 4),
                 intra_layers: int = 1, inter_layers: int = 2,
                 label_to_anatomy: Optional[Mapping[str, Sequence[int]]] = None,
                 predict_missing: bool = False, temporal_size: int = 5):
        super().__init__()
        if tuple(source_order) != VIEWS:
            raise ValueError(f'source_order must be {VIEWS}')
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        if inter_layers != 2:
            raise ValueError('inter_layers must be 2 for LAX then SAX cross-attention')
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
        self.temporal_size = temporal_size
        self.encoders = nn.ModuleDict({
            v: (HybridViewEncoder(in_channels, base, dim, token_grid, intra_layers, heads)
                if v == 'sa' else CineViewEncoder(in_channels, base, dim, token_grid,
                                                  intra_layers, heads, temporal_size))
            for v in VIEWS})
        self.modality_embedding = nn.Parameter(torch.randn(3, 1, dim) * .02)
        self.inter = TwoStageAnatomyTransformer(
            dim, heads, torch.stack([getattr(self, f'anatomy_available_{v}') for v in VIEWS]),
            inter_layers)
        self.context_scale = nn.Parameter(torch.tensor(.1))
        token_counts = {v: int(getattr(self, f'anatomy_available_{v}').sum()) for v in VIEWS}
        self.decoder = AnatomyAwareProgressiveDecoder(base, dim, {'sa': token_counts['sa']})
        self.decoder_2d = AnatomyAwareProgressiveDecoder(
            base, dim, {v: token_counts[v] for v in VIEWS[:2]}, spatial_dims=2)
        # LAX heads predict only the center frame.
        output_channels = dict(self.num_classes_by_source)
        self.heads = nn.ModuleDict({v: (nn.Conv3d if v == 'sa' else nn.Conv2d)(base, output_channels[v], 1)
                                   for v in VIEWS})
        self.aux_decoder = ProgressiveDecoder(base, dim)
        self.aux_decoder_2d = ProgressiveDecoder(base, dim, spatial_dims=2)
        self.aux_heads = nn.ModuleDict({v: (nn.Conv3d if v == 'sa' else nn.Conv2d)(base, output_channels[v], 1)
                                       for v in VIEWS})
    def forward(self, x: Mapping[str, Tensor], modality_mask: Tensor) -> Dict:
        """LAX: [B,C,T,H,W] or [B,C*T,H,W]; SAX: [B,C,D,H,W].

        LAX logits return [B,classes,H,W] for the window center, including auxiliary heads.
        The temporal length must match the configured temporal_size (default 5).
        """
        if not isinstance(x, Mapping) or set(x) != set(VIEWS):
            raise ValueError('x must be a mapping with 2ch, 4ch and sa tensors')
        x = dict(x)
        for view in VIEWS[:2]:
            image = x[view]
            if image.ndim == 4:
                if image.shape[1] != self.in_channels * self.temporal_size:
                    raise ValueError(f'{view} expects {self.in_channels * self.temporal_size} input channels')
                x[view] = image.reshape(image.shape[0], self.in_channels, self.temporal_size,
                                        *image.shape[-2:])
        reference = x['sa']
        batch = reference.shape[0]
        for view in VIEWS:
            if x[view].ndim != 5 or x[view].shape[:2] != (batch, self.in_channels):
                raise ValueError(f'{view} expects [B,C,' + ('D,H,W]' if view == 'sa' else 'T,H,W]'))
            if view != 'sa' and x[view].shape[2] != self.temporal_size:
                raise ValueError(f'{view} expects T={self.temporal_size}, got T={x[view].shape[2]}')
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
            shapes = [tuple(image.shape[2:] if view == 'sa' else image.shape[-2:])]
            for _ in range(3):
                previous = shapes[-1]
                shapes.append(tuple((n + 1) // 2 for n in previous))
            spatial_tokens = tuple((n + 1) // 2 for n in shapes[-1])
            token_shape = spatial_tokens if view == 'sa' else (self.temporal_size, *spatial_tokens)
            patch_shapes[view] = token_shape
            patches = prod(token_shape)
            indices = present[:, i].nonzero(as_tuple=True)[0]
            if indices.numel():
                sk, tok, tokshape = self.encoders[view](image.index_select(0, indices))
                if tuple(tokshape) != token_shape:
                    raise RuntimeError('Token grid mismatch')
                skips = tuple(f.new_zeros((batch, *f.shape[1:])).index_copy(0, indices, f)
                              for f in sk)
                tokfull = tok.new_zeros((batch, patches, self.dim)).index_copy(0, indices, tok)
            else:
                skips = tuple(image.new_zeros((batch, c, *sh)) for c, sh in zip(chans, shapes))
                tokfull = image.new_zeros((batch, patches, self.dim))
            skip_maps[view] = skips
            view_tokens[view] = tokfull + self.modality_embedding[i] * present[:, i, None, None]
        fused, anatomy_tokens = self.inter(view_tokens, present)
        logits, aux_logits = {}, {}
        for i, view in enumerate(VIEWS):
            indices = present[:, i].nonzero(as_tuple=True)[0]
            nclasses = self.num_classes_by_source[view]
            output_shape = (batch, nclasses, *(x[view].shape[2:] if view == 'sa' else x[view].shape[-2:]))
            if not indices.numel():
                logits[view] = reference.new_zeros(output_shape)
                if self.training:
                    aux_logits[view] = reference.new_zeros(output_shape)
                continue
            is_2d = view != 'sa'
            mode = 'bilinear' if is_2d else 'trilinear'
            decoder = self.decoder_2d if is_2d else self.decoder
            aux_decoder = self.aux_decoder_2d if is_2d else self.aux_decoder
            sk = [f.index_select(0, indices) for f in skip_maps[view]]
            patch = fused[view].index_select(0, indices).transpose(1, 2)
            patch = patch.reshape(indices.numel(), self.dim, *patch_shapes[view])
            if is_2d:
                patch = patch[:, :, self.temporal_size // 2]
            patch = F.interpolate(patch, size=sk[-1].shape[2:], mode=mode, align_corners=False)
            sk[-1] = sk[-1] + self.context_scale * patch
            anatomy = anatomy_tokens.index_select(0, indices)
            d1 = decoder(sk, anatomy,
                                  getattr(self, f'anatomy_available_{view}'), view)

            def restore(pred):
                size = output_shape[-2:] if is_2d else output_shape[2:]
                pred = F.interpolate(pred, size=size, mode=mode, align_corners=False)
                return pred.new_zeros(output_shape).index_copy(0, indices, pred)

            logits[view] = restore(self.heads[view](d1))
            if self.training:
                aux1 = aux_decoder([f.index_select(0, indices) for f in skip_maps[view]])
                aux_logits[view] = restore(self.aux_heads[view](aux1))
        return dict(logits=logits, aux_logits=aux_logits,
                    pred_tokens=fused,
                    anatomy_tokens=anatomy_tokens, valid_views=present, patch_shapes=patch_shapes)
