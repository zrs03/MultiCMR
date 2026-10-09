from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 基础模块
# ----------------------------------------------------------------------

class AnatomyMaskedTransformer(nn.Module):
    """
    基于解剖可见性掩码（Visibility Attention Mask）的多切面 Vision Transformer。
    纯序列注意力机制：控制各切面 Patch 与不同解剖 Token 之间的可见性。
    """
    def __init__(
        self,
        feature_dim: int = 256,
        num_patches_per_view: int = 8,
        num_anatomy_nodes: int = 5,
        heads: int = 8,
        layers: int = 2,
    ):
        super().__init__()
        self.num_patches = num_patches_per_view
        self.num_anatomy = num_anatomy_nodes
        self.total_tokens = 3 * num_patches_per_view + num_anatomy_nodes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.patch_pos_embedding = nn.Parameter(
            torch.randn(1, 1, num_patches_per_view, feature_dim) * 0.02
        )
        self.anatomy_pos_embedding = nn.Parameter(
            torch.randn(1, num_anatomy_nodes, feature_dim) * 0.02
        )

        self.register_buffer(
            "attention_mask",
            self.build_anatomy_attention_mask(),
        )

    def build_anatomy_attention_mask(self):
        P = self.num_patches
        N = self.total_tokens

        mask = torch.full((N, N), float("-inf"))
        mask.fill_diagonal_(0.0)
        mask[: 3 * P, : 3 * P] = 0.0

        view_slices = {
            "2ch": slice(0 * P, 1 * P),
            "4ch": slice(1 * P, 2 * P),
            "sa":  slice(2 * P, 3 * P),
        }

        anatomy_visibility = [
            (0, ["2ch", "4ch", "sa"]),  # LV_myo
            (1, ["2ch", "4ch", "sa"]),  # LV_cav
            (2, ["4ch", "sa"]),         # RV_cav
            (3, ["4ch"]),               # RA
            (4, ["4ch"]),               # LA
        ]

        anat_base = 3 * P
        for anat_idx, visible_views in anatomy_visibility:
            token_pos = anat_base + anat_idx
            for view in visible_views:
                v_slice = view_slices[view]
                mask[v_slice, token_pos] = 0.0
                mask[token_pos, v_slice] = 0.0

        return mask

    def forward(self, nodes, key_padding_mask=None):
        P = self.num_patches
        B, N, C = nodes.shape

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, N):
                raise ValueError(
                    f"key_padding_mask shape must be {(B, N)}, "
                    f"but got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(
                device=nodes.device,
                dtype=torch.bool,
            )

        patch_nodes = nodes[:, : 3 * P, :].view(B, 3, P, C)
        patch_nodes = patch_nodes + self.patch_pos_embedding
        patch_nodes = patch_nodes.view(B, 3 * P, C)

        anatomy_nodes = nodes[:, 3 * P :, :]
        anatomy_nodes = anatomy_nodes + self.anatomy_pos_embedding

        nodes = torch.cat([patch_nodes, anatomy_nodes], dim=1)

        return self.transformer(
            nodes,
            mask=self.attention_mask.to(device=nodes.device),
            src_key_padding_mask=key_padding_mask,
        )


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _, _ = x.shape
        pooled = self.pool(x).view(batch_size, channels)
        weights = torch.relu(self.fc1(pooled))
        weights = torch.sigmoid(self.fc2(weights)).view(batch_size, channels, 1, 1, 1)
        return x * weights


class StemBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels),
        )
        self.se = SqueezeExcitation3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv_path(x) + self.skip_path(x))


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels),
        )
        self.se = SqueezeExcitation3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv_path(x) + self.skip_path(x))


class ASPP3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: Sequence[int] = (1, 2, 4, 6)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
            for rate in rates
        ])
        self.proj = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = None
        for branch in self.branches:
            out = branch(x)
            fused = out if fused is None else fused + out
        return self.proj(fused)


class AttentionGate3D(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.g_proj = nn.Sequential(
            nn.BatchNorm3d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(gate_channels, skip_channels, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.x_proj = nn.Sequential(
            nn.BatchNorm3d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels, skip_channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            nn.BatchNorm3d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels, skip_channels, kernel_size=3, padding=1),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gated = self.g_proj(gate)
        skip_proj = self.x_proj(skip)
        if gated.shape[2:] != skip_proj.shape[2:]:
            gated = F.interpolate(gated, size=skip_proj.shape[2:], mode="trilinear", align_corners=False)
        out = self.out_conv(gated + skip_proj)
        return out * skip


class DecoderBlock3D(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = AttentionGate3D(gate_channels, skip_channels)
        self.res = ResBlock3D(gate_channels + skip_channels, out_channels, stride=1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attn_skip = self.attn(gate, skip)
        gate_up = F.interpolate(gate, size=attn_skip.shape[2:], mode="trilinear", align_corners=False)
        out = torch.cat([attn_skip, gate_up], dim=1)
        return self.res(out)


# ----------------------------------------------------------------------
# 独立的单视图 Encoder
# ----------------------------------------------------------------------

class Encoder3D(nn.Module):
    """
    单视图 3D 编码器。
    每个视图各持有一份实例，低层特征与 BN 统计量完全按本视图分布学习。
    """
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.c1 = StemBlock3D(in_channels, 32, stride=2)
        self.c2 = ResBlock3D(32, 32, 2)
        self.c3 = ResBlock3D(32, 64, 2)
        self.b = ASPP3D(64, 256)

    def forward(self, x: torch.Tensor):
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        b = self.b(s3)
        return s1, s2, s3, b


# ----------------------------------------------------------------------
# 共享 Decoder 主干
# ----------------------------------------------------------------------

class SharedDecoderTrunk3D(nn.Module):
    """
    三视图共享的解码主干：d1~d3 + ASPP，输出 16 通道特征图。
    分割解码规则跨视图共通，共享主干等价于跨视图正则化。
    FiLM 层共享，视图差异由 256 维解剖条件向量 cond 的内容表达。
    类别数不同（2ch:3 / 4ch:6 / sa:4），分割 head 不共享，
    见 ResUNetPP3DMultiHead.heads。
    """
    def __init__(self) -> None:
        super().__init__()
        self.d1 = DecoderBlock3D(gate_channels=256, skip_channels=64, out_channels=128)
        self.d2 = DecoderBlock3D(gate_channels=128, skip_channels=32, out_channels=64)
        self.d3 = DecoderBlock3D(gate_channels=64, skip_channels=32, out_channels=32)
        self.aspp = ASPP3D(32, 16)

        self.film_d1 = nn.Linear(256, 128 * 2)
        self.film_d2 = nn.Linear(256, 64 * 2)
        self.film_d3 = nn.Linear(256, 32 * 2)

        nn.init.zeros_(self.film_d1.weight)
        nn.init.zeros_(self.film_d1.bias)
        nn.init.zeros_(self.film_d2.weight)
        nn.init.zeros_(self.film_d2.bias)
        nn.init.zeros_(self.film_d3.weight)
        nn.init.zeros_(self.film_d3.bias)

    def apply_film(self, x: torch.Tensor, film_layer: nn.Module, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = film_layer(cond)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return x * (1.0 + gamma) + beta

    def forward(self, skip1, skip2, skip3, bottleneck, cond=None):
        """
        return: [B, 16, D, H, W] 共享解码特征，交由各视图独立 head 分类。
        """
        x = self.d1(bottleneck, skip3)
        if cond is not None:
            x = self.apply_film(x, self.film_d1, cond)

        x = self.d2(x, skip2)
        if cond is not None:
            x = self.apply_film(x, self.film_d2, cond)

        x = self.d3(x, skip1)
        if cond is not None:
            x = self.apply_film(x, self.film_d3, cond)

        return self.aspp(x)


# ----------------------------------------------------------------------
# MAE 式缺失视图重建器
# ----------------------------------------------------------------------

class MissingViewReconstructor(nn.Module):
    """
    MAE 风格的缺失视图 patch 重建。

    P 个可学习 query（mask token + 位置嵌入）通过 cross-attention
    从 memory（存在视图的 patch tokens + sample-adaptive anatomy tokens）
    中逐 patch 取信息。重建结果随样本变化，参数量与 P 无关。
    """
    def __init__(
        self,
        num_patches: int,
        feature_dim: int = 256,
        heads: int = 8,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches

        self.mask_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = memory.shape[0]
        queries = (self.mask_token + self.query_pos).expand(B, -1, -1)
        out = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_norm(out)


# ----------------------------------------------------------------------
# 主模型：三 Encoder + 共享 Decoder
# ----------------------------------------------------------------------

class ResUNetPP3DMultiHead(nn.Module):
    def __init__(self, in_channels=1, source_order=("2ch", "4ch", "sa"), num_classes_by_source=None):
        super().__init__()
        self.source_order = list(source_order)
        self.num_classes_by_source = num_classes_by_source or {"2ch": 3, "4ch": 6, "sa": 4}

        self.patch_size = (2, 2, 2)
        self.patch_grid = (4, 10, 10)
        self.num_patches = self.patch_grid[0] * self.patch_grid[1] * self.patch_grid[2]

        self.encoders = nn.ModuleDict({
            src: Encoder3D(in_channels) for src in self.source_order
        })

        # 每视图独立的 patch 投影：把本视图 encoder 的 bottleneck 特征
        # 映射到 token 空间，投影分布按本视图学习。
        self.patch_projs = nn.ModuleDict({
            src: nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            for src in self.source_order
        })

        # token 层 modality embedding：颈部融合序列需要区分 patch 来自哪个视图。
        self.modality_embedding = nn.ParameterDict({
            "2ch": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
            "4ch": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
            "sa": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
        })

        # 5 个解剖结构的 [CLS] Tokens (LV_myo, LV_cav, RV_cav, RA, LA)
        self.anatomy_tokens = nn.Parameter(torch.randn(5, 256) * 0.02)

        # 第一阶段：只利用当前存在的模态更新 anatomy tokens。
        self.anatomy_update_neck = AnatomyMaskedTransformer(
            feature_dim=256,
            num_patches_per_view=self.num_patches,
            num_anatomy_nodes=5,
            heads=8,
            layers=2,
        )

        # 第二阶段：更新后的 anatomy tokens + 重建后的 patch tokens 做最终跨模态融合。
        self.transformer_neck = AnatomyMaskedTransformer(
            feature_dim=256,
            num_patches_per_view=self.num_patches,
            num_anatomy_nodes=5,
            heads=8,
            layers=2,
        )

        self.anatomy_indices_by_view = {
            "2ch": [0, 1],               # LV_myo, LV_cav
            "4ch": [0, 1, 2, 3, 4],      # 全部 5 个结构
            "sa": [0, 1, 2],             # LV_myo, LV_cav, RV_cav
        }

        self.reconstructors = nn.ModuleDict({
            src: MissingViewReconstructor(
                num_patches=self.num_patches,
                feature_dim=256,
                heads=8,
                layers=2,
            )
            for src in self.source_order
        })

        # 将连通的解剖 CLS Token 聚合映射为当前切面的条件向量 cond [B, 256]
        # 各视图连通的解剖结构数量不同，cond 投影保持按视图独立。
        self.anatomy_cond_proj = nn.ModuleDict({
            src: nn.Sequential(
                nn.Linear(len(self.anatomy_indices_by_view[src]) * 256, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
            )
            for src in self.source_order
        })

        self.gamma_patch = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma_recon = nn.Parameter(torch.ones(1))

        self.decoder_trunk = SharedDecoderTrunk3D()
        self.heads = nn.ModuleDict({
            src: nn.Conv3d(16, self.num_classes_by_source[src], kernel_size=1)
            for src in self.source_order
        })

    def encode(self, src: str, x: torch.Tensor):
        """用视图 src 自己的 encoder 编码。"""
        return self.encoders[src](x)

    def feature_to_token(self, src: str, b):
        x_proj = self.patch_projs[src](b)
        B, C, _, _, _ = x_proj.shape
        return x_proj.view(B, C, -1).transpose(1, 2)

    def forward(self, x, modality_mask, full_x=None):
        """
        x: [B, 3, D, H, W] (masked inputs)
        modality_mask: [B, 3]
        full_x: [B, 3, D, H, W] (unmasked full inputs, optional)
        """
        B = x.shape[0]
        P = self.num_patches
        modality_features = []
        skips = {}

        # gt_tokens：用各视图自己的 encoder 编码完整输入（no_grad 稳定监督）。
        # loss 端须用 modality_mask 屏蔽 missing 视图（见 token_distillation_loss）。
        gt_tokens = {}
        if full_x is not None:
            with torch.no_grad():
                for i, src in enumerate(self.source_order):
                    xi_full = full_x[:, i : i + 1]
                    _, _, _, b_full = self.encode(src, xi_full)
                    gt_token = self.feature_to_token(src, b_full) + self.modality_embedding[src]
                    gt_tokens[src] = gt_token

        anatomy_nodes_init = self.anatomy_tokens.unsqueeze(0).expand(B, -1, -1)

        raw_modality_tokens = []
        modality_present = []

        for i, src in enumerate(self.source_order):
            present = modality_mask[:, i].bool()
            xi = x[:, i : i + 1]

            s1, s2, s3, b = self.encode(src, xi)
            skips[src] = (s1, s2, s3, b)

            token = self.feature_to_token(src, b) + self.modality_embedding[src]
            raw_modality_tokens.append(token)
            modality_present.append(present)

        raw_modality_nodes = torch.cat(raw_modality_tokens, dim=1)

        # ------------------------------------------------------------------
        # 第一阶段：从当前样本实际存在的模态更新 anatomy tokens。
        # ------------------------------------------------------------------
        anatomy_context_input = torch.cat(
            [raw_modality_nodes, anatomy_nodes_init],
            dim=1,
        )

        anatomy_key_padding_mask = torch.zeros(
            B,
            3 * P + self.transformer_neck.num_anatomy,
            dtype=torch.bool,
            device=x.device,
        )
        for i, present in enumerate(modality_present):
            missing_patch_mask = (~present).unsqueeze(1).expand(B, P)
            start = i * P
            end = (i + 1) * P
            anatomy_key_padding_mask[:, start:end] = missing_patch_mask

        anatomy_context_out = self.anatomy_update_neck(
            anatomy_context_input,
            key_padding_mask=anatomy_key_padding_mask,
        )
        anatomy_nodes = anatomy_context_out[:, 3 * P :, :]  # [B, 5, 256]

        # ------------------------------------------------------------------
        # 第二阶段：MAE 式缺失视图重建
        # ------------------------------------------------------------------
        recon_memory = torch.cat([raw_modality_nodes, anatomy_nodes], dim=1)
        recon_memory_kpm = anatomy_key_padding_mask

        modality_features = []
        for i, src in enumerate(self.source_order):
            present = modality_present[i]
            token = raw_modality_tokens[i].clone()
            missing = ~present

            if missing.any():
                recon = self.reconstructors[src](
                    recon_memory,
                    memory_key_padding_mask=recon_memory_kpm,
                )
                token[missing] = self.gamma_recon * recon[missing]

            modality_features.append(token)

        modality_nodes = torch.cat(modality_features, dim=1)

        # ------------------------------------------------------------------
        # 最终跨模态 Transformer 融合。
        # ------------------------------------------------------------------
        tokens = torch.cat([modality_nodes, anatomy_nodes], dim=1)
        neck_out = self.transformer_neck(tokens)

        updated_patches = neck_out[:, : 3 * P, :].view(B, 3, P, 256)
        updated_anatomy = neck_out[:, 3 * P :, :]

        outputs = {}
        pred_tokens = {}
        for i, src in enumerate(self.source_order):
            s1, s2, s3, b = skips[src]
            target_shape = b.shape[2:]

            view_patches = updated_patches[:, i]
            pred_tokens[src] = view_patches

            patch_feat = view_patches.transpose(1, 2).view(B, 256, *self.patch_grid)
            patch_context_feat = F.interpolate(patch_feat, size=target_shape, mode="trilinear", align_corners=False)
            b = b + self.gamma_patch * patch_context_feat

            sel_indices = self.anatomy_indices_by_view[src]
            sel_tokens = updated_anatomy[:, sel_indices, :]
            cond = self.anatomy_cond_proj[src](sel_tokens.reshape(B, -1))

            feat = self.decoder_trunk(s1, s2, s3, b, cond=cond)
            outputs[src] = self.heads[src](feat)

        return {
            "logits": outputs,
            "pred_tokens": pred_tokens,
            "gt_tokens": gt_tokens,
        }
