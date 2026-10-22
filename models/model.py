"""Temporal LAX and spatiotemporal 2.5D SAX with sequential anatomy cross-attention.
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
from .mednext import MedNeXtEncoder, MedNeXtDecoder
VIEWS = ("2ch", "4ch", "sa")
ANATOMY = ("background", "LV_myo", "LV_cav", "RV_cav", "RA", "LA")
DEFAULT_LABEL_MAP = {"2ch": (0, 1, 2), "4ch": (0, 1, 2, 3, 4, 5), "sa": (0, 1, 2, 3)}


def norm(c: int) -> nn.Module:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


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


class CineViewEncoder(nn.Module):
    """Five context channels -> complete 2D MedNeXt-M -> spatial tokens."""
    def __init__(self, in_channels, base, dim, token_grid, intra_layers, heads, temporal_size=5):
        super().__init__()
        if not isinstance(temporal_size, int) or temporal_size < 1 or temporal_size % 2 == 0:
            raise ValueError('temporal_size must be a positive odd integer')
        self.in_channels, self.temporal_size = in_channels, temporal_size
        self.backbone = MedNeXtEncoder(in_channels * temporal_size, base)
        self.patch_embed = nn.Conv2d(base * 16, dim, 1)
        self.token_pos = nn.Parameter(torch.randn(1, dim, *token_grid[-2:]) * .02)
        self.intra = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(intra_layers)])
        self.token_norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 5:
            if x.shape[1:3] != (self.in_channels, self.temporal_size):
                raise ValueError(f'Cine expects C={self.in_channels}, T={self.temporal_size}; got {tuple(x.shape)}')
            x = x.flatten(1, 2)
        if x.ndim != 4 or x.shape[1] != self.in_channels * self.temporal_size:
            raise ValueError('Cine expects [B,C*K,H,W] or [B,C,K,H,W]')
        features = self.backbone(x)
        patches = self.patch_embed(features[-1])
        shape = patches.shape[-2:]
        pos = F.interpolate(self.token_pos, size=shape, mode='bilinear', align_corners=False)
        tokens = (patches + pos).flatten(2).transpose(1, 2)
        for layer in self.intra:
            tokens = layer(tokens)
        return features, self.token_norm(tokens), shape


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

    def forward(self, patches, anatomy, allowed, read_allowed=None):
        # Token dropout only restricts anatomy reads, never patch queries.
        read_allowed = allowed if read_allowed is None else read_allowed
        anatomy = self.anatomy_reads(anatomy, patches, read_allowed.transpose(1, 2))
        patches = self.patch_reads(patches, anatomy, allowed)
        return patches, anatomy


class TwoStageAnatomyTransformer(nn.Module):
    """Two routed cross-attention stages followed by one joint self-attention block."""
    def __init__(self, dim: int, heads: int, view_anatomy: Tensor, layers: int = 2,
                 dropout: float = .1, token_drop_rate: float = .05):
        super().__init__()
        if not 0 <= token_drop_rate < 1:
            raise ValueError("token_drop_rate must be in [0, 1)")
        self.token_drop_rate = token_drop_rate
        if layers != 2:
            raise ValueError('inter_layers must be 2: one LAX stage, then one SAX stage')
        self.register_buffer('view_anatomy', view_anatomy.to(dtype=torch.bool).clone())
        self.anatomy_tokens = nn.Parameter(torch.randn(1, 5, dim) * .02)
        self.lax_patch_attention = TransformerBlock(dim, heads, dropout)
        self.lax_stage = AnatomyCrossStage(dim, heads, dropout)
        self.sax_stage = AnatomyCrossStage(dim, heads, dropout)
        self.joint_attention = TransformerBlock(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def mask_anatomy_reads(self, edges, present):
        if not self.training or self.token_drop_rate == 0:
            return edges
        eligible = present.sum(1) >= 2
        masked = []
        for i, allowed in enumerate(edges):
            batch, length, _ = allowed.shape
            # Round to the nearest token; tiny grids may have zero dropped tokens.
            count = min(length - 1, int(length * self.token_drop_rate + .5))
            if count == 0:
                masked.append(allowed)
                continue
            keep = torch.ones(batch, length, dtype=torch.bool, device=allowed.device)
            indices = torch.rand(batch, length, device=allowed.device).topk(count, dim=1).indices
            keep.scatter_(1, indices, False)
            keep |= ~(eligible & present[:, i])[:, None]
            masked.append(allowed & keep[:, :, None])
        return masked

    def forward(self, tokens: Mapping[str, Tensor], present: Tensor):
        batch = tokens['sa'].shape[0]
        if present.shape != (batch, 3) or not present.bool().any(dim=1).all():
            raise ValueError('Expected nonempty present mask [B,3]')
        present = present.bool()
        patches = [tokens[v].masked_fill(~present[:, i, None, None], 0)
                   for i, v in enumerate(VIEWS)]
        edges = [self.view_anatomy[i][None, None, :].expand(batch, p.shape[1], -1)
                 & present[:, i, None, None] for i, p in enumerate(patches)]
        # Joint LAX self-attention: both views can directly read each other's patches.
        # Exclude SAX-only rows so attention never receives an all-masked key set.
        lax = torch.cat(patches[:2], 1)
        missing = torch.cat([~present[:, i:i+1].expand(-1, patches[i].shape[1])
                             for i in range(2)], 1)
        indices = present[:, :2].any(1).nonzero(as_tuple=True)[0]
        if indices.numel():
            updated_lax = self.lax_patch_attention(
                lax.index_select(0, indices),
                key_padding_mask=missing.index_select(0, indices))
            lax = lax.index_copy(0, indices, updated_lax)
        lax = lax.masked_fill(missing[:, :, None], 0)
        read_edges = self.mask_anatomy_reads(edges, present)
        anatomy = self.anatomy_tokens.expand(batch, -1, -1)
        lax, anatomy = self.lax_stage(lax, anatomy,
                                      torch.cat(edges[:2], 1), torch.cat(read_edges[:2], 1))
        sax, anatomy = self.sax_stage(patches[2], anatomy, edges[2], read_edges[2])
        # Stage three: all present spatial tokens and five anatomy tokens interact.
        # Anatomy keys remain valid even for single-view inputs. Missing patch
        # queries are discarded below and cannot serve as keys or values.
        joint_missing = torch.cat([
            ~present[:, i:i+1].expand(-1, p.shape[1])
            for i, p in enumerate(patches)
        ] + [torch.zeros(batch, 5, dtype=torch.bool, device=present.device)], dim=1)
        joint = torch.cat((lax, sax, anatomy), dim=1)
        joint = joint.masked_fill(joint_missing[:, :, None], 0)
        joint = self.joint_attention(joint, key_padding_mask=joint_missing)
        parts = joint.split([p.shape[1] for p in patches] + [5], dim=1)
        updated, anatomy = parts[:3], parts[3]
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


class AnatomyAwareProgressiveDecoder(MedNeXtDecoder):
    """Four MedNeXt decoding stages, each followed by per-view FiLM."""
    def __init__(self, base, dim, view_token_counts):
        super().__init__(base)
        for scale in (1,2,3,4):
            channels = base * 2**(scale-1)
            setattr(self, f'a{scale}', nn.ModuleDict({
                v: AnatomyFiLM(channels, dim, n) for v,n in view_token_counts.items()}))
        self.spatial_adapters = nn.ModuleDict({
            v: nn.ModuleList([ViewSpatialAdapter(base * m) for m in (8,4,2,1)])
            for v in view_token_counts if v != 'sa'})

    def forward(self, skips, anatomy, availability, view):
        x = skips[-1]
        decoded = []
        for i, skip in enumerate(reversed(skips[:-1])):
            x = self.decode_stage(i, x, skip)
            x = getattr(self, f'a{4-i}')[view](x, anatomy, availability)
            if view in self.spatial_adapters:
                x = self.spatial_adapters[view][i](x)
            decoded.append(x)
        return decoded[-1], decoded[-2]


class MMFormerAnatomy(nn.Module):
    """LAX/anatomy then SAX/anatomy cross-attention and decoder FiLM."""
    def __init__(self, in_channels: int = 1,
                 num_classes_by_source: Optional[Dict[str, int]] = None,
                 source_order: Sequence[str] = VIEWS,
                 base: int = 32, dim: int = 128, heads: int = 4,
                 token_grid: Tuple[int, int, int] = (2, 4, 4),
                 intra_layers: int = 1, inter_layers: int = 2,
                 label_to_anatomy: Optional[Mapping[str, Sequence[int]]] = None,
                 predict_missing: bool = False, temporal_size: int = 5,
                 token_drop_rate: float = .05):
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
            v: CineViewEncoder(in_channels, base, dim, token_grid, intra_layers, heads,
                               5 if v == 'sa' else temporal_size)
            for v in VIEWS})
        self.modality_embedding = nn.Parameter(torch.randn(3, 1, dim) * .02)
        self.inter = TwoStageAnatomyTransformer(
            dim, heads, torch.stack([getattr(self, f'anatomy_available_{v}') for v in VIEWS]),
            inter_layers, token_drop_rate=token_drop_rate)
        self.context_projection = nn.ModuleDict({v: nn.Conv2d(dim, base * 16, 1) for v in VIEWS})
        self.context_scale = nn.Parameter(torch.tensor(.1))
        token_counts = {v: int(getattr(self, f'anatomy_available_{v}').sum()) for v in VIEWS}
        self.decoder = AnatomyAwareProgressiveDecoder(base, dim, {'sa': token_counts['sa']})
        self.decoder_2d = AnatomyAwareProgressiveDecoder(
            base, dim, {v: token_counts[v] for v in VIEWS[:2]})
        # LAX heads predict only the center frame.
        output_channels = dict(self.num_classes_by_source)
        self.heads = nn.ModuleDict({v: nn.Conv2d(base, output_channels[v], 1)
                                   for v in VIEWS})
        self.deep_heads = nn.ModuleDict({
            v: nn.Conv2d(base * 2, output_channels[v], 1)
            for v in VIEWS})
        self.aux_decoder = MedNeXtDecoder(base)
        self.aux_decoder_2d = MedNeXtDecoder(base)
        self.aux_heads = nn.ModuleDict({v: nn.Conv2d(base, output_channels[v], 1)
                                       for v in VIEWS})
    def forward(self, x: Mapping[str, Tensor], modality_mask: Tensor) -> Dict:
        """LAX: [B,C,T,H,W]; SAX: [B,C,5,H,W] (center,t-,t+,z-,z+).

        All view logits return [B,classes,H,W] for the window center, including auxiliary heads.
        The temporal length must match the configured temporal_size (default 5).
        """
        if not isinstance(x, Mapping) or set(x) != set(VIEWS):
            raise ValueError('x must be a mapping with 2ch, 4ch and sa tensors')
        x = dict(x)
        for view in VIEWS:
            channels = 5 if view == 'sa' else self.temporal_size
            image = x[view]
            if image.ndim == 4:
                if image.shape[1] != self.in_channels * channels:
                    raise ValueError(f'{view} expects {self.in_channels * channels} input channels')
                x[view] = image.reshape(image.shape[0], self.in_channels, channels, *image.shape[-2:])
        reference = x['sa']
        batch = reference.shape[0]
        for view in VIEWS:
            channels = 5 if view == 'sa' else self.temporal_size
            if x[view].ndim != 5 or x[view].shape[:2] != (batch, self.in_channels):
                raise ValueError(f'{view} expects [B,C,K,H,W]')
            if x[view].shape[2] != channels:
                raise ValueError(f'{view} expects T={channels}, got T={x[view].shape[2]}')
            if x[view].device != reference.device:
                raise ValueError('All views must be on the same device')
        if modality_mask.shape != (batch, 3):
            raise ValueError('modality_mask must have shape [B,3]')
        present = modality_mask.to(device=reference.device, dtype=torch.bool)
        if not present.any(dim=1).all():
            raise ValueError('Each sample must have at least one available view')
        skip_maps, view_tokens, patch_shapes = {}, {}, {}
        chans = tuple(self.base * 2**i for i in range(5))
        for i, view in enumerate(VIEWS):
            image = x[view]
            shapes = [tuple(image.shape[-2:])]
            for _ in range(4):
                previous = shapes[-1]
                shapes.append(tuple((n + 1) // 2 for n in previous))
            token_shape = shapes[-1]
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
        logits, aux_logits, deep_logits = {}, {}, {}
        for i, view in enumerate(VIEWS):
            indices = present[:, i].nonzero(as_tuple=True)[0]
            nclasses = self.num_classes_by_source[view]
            output_shape = (batch, nclasses, *(x[view].shape[-2:]))
            if not indices.numel():
                logits[view] = reference.new_zeros(output_shape)
                if self.training:
                    aux_logits[view] = reference.new_zeros(output_shape)
                    deep_logits[view] = reference.new_zeros(output_shape)
                continue
            mode = 'bilinear'
            decoder = self.decoder if view == 'sa' else self.decoder_2d
            aux_decoder = self.aux_decoder if view == 'sa' else self.aux_decoder_2d
            sk = [f.index_select(0, indices) for f in skip_maps[view]]
            patch = fused[view].index_select(0, indices).transpose(1, 2)
            patch = patch.reshape(indices.numel(), self.dim, *patch_shapes[view])
            patch = F.interpolate(patch, size=sk[-1].shape[2:], mode=mode, align_corners=False)
            sk[-1] = sk[-1] + self.context_scale * self.context_projection[view](patch)
            anatomy = anatomy_tokens.index_select(0, indices)
            d1, d2 = decoder(sk, anatomy,
                                  getattr(self, f'anatomy_available_{view}'), view)

            def restore(pred):
                size = output_shape[-2:]
                pred = F.interpolate(pred, size=size, mode=mode, align_corners=False)
                return pred.new_zeros(output_shape).index_copy(0, indices, pred)

            logits[view] = restore(self.heads[view](d1))
            if self.training:
                deep_logits[view] = restore(self.deep_heads[view](d2))
                aux1 = aux_decoder([f.index_select(0, indices) for f in skip_maps[view]])
                aux_logits[view] = restore(self.aux_heads[view](aux1))
        return dict(logits=logits, aux_logits=aux_logits, deep_logits=deep_logits,
                    pred_tokens=fused,
                    anatomy_tokens=anatomy_tokens, valid_views=present, patch_shapes=patch_shapes)
