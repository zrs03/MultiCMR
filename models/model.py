from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        纯 Transformer 块掩码（Block-Masking）构造逻辑：
        序列排布: [2CH_patches (P), 4CH_patches (P), SA_patches (P), Anatomy_Tokens (5)]
        
        Anatomy 索引:
            0: LV_myo
            1: LV_cav
            2: RV_cav
            3: RA
            4: LA
        """
        P = self.num_patches
        N = self.total_tokens
        
        # 默认全部不可见 (-inf)
        mask = torch.full((N, N), float("-inf"))

        # 1. 对角线自身可见
        mask.fill_diagonal_(0.0)

        # 2. Patch-Patch: 只允许同视图内部互通，禁止跨切面 Patch 直接互看
        # View 0 (2CH): 0 ~ P-1
        # View 1 (4CH): P ~ 2P-1
        # View 2 (SA):  2P ~ 3P-1
        for v in range(3):
            v_start = v * P
            v_end = (v + 1) * P
            mask[v_start:v_end, v_start:v_end] = 0.0

        # 3. Patch-Anatomy: 保持生理连通
        view_slices = {
            "2ch": slice(0 * P, 1 * P),
            "4ch": slice(1 * P, 2 * P),
            "sa":  slice(2 * P, 3 * P),
        }

        anatomy_visibility = [
            (0, ["2ch", "4ch", "sa"]),  # LV_myo <-> 2CH, 4CH, SA
            (1, ["2ch", "4ch", "sa"]),  # LV_cav <-> 2CH, 4CH, SA
            (2, ["4ch", "sa"]),         # RV_cav <-> 4CH, SA
            (3, ["4ch"]),               # RA     <-> 4CH
            (4, ["4ch"]),               # LA     <-> 4CH
        ]

        anat_base = 3 * P
        for anat_idx, visible_views in anatomy_visibility:
            token_pos = anat_base + anat_idx
            for view in visible_views:
                v_slice = view_slices[view]
                # 双向可见: View Patch <-> Anatomy Token
                mask[v_slice, token_pos] = 0.0
                mask[token_pos, v_slice] = 0.0

        # 4. Anatomy-Anatomy: 增加生理拓扑连接
        # 生理邻接与血流拓扑边关系:
        # (0, 1): LV_myo <-> LV_cav (心肌包绕心室)
        # (0, 2): LV_myo <-> RV_cav (室间隔与右室相连)
        # (1, 2): LV_cav <-> RV_cav (左右心室通过室间隔相邻)
        # (1, 4): LV_cav <-> LA     (二尖瓣连通)
        # (2, 3): RV_cav <-> RA     (三尖瓣连通)
        # (3, 4): RA     <-> LA     (房间隔相邻)
        anatomy_topo_pairs = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 4),
            (2, 3),
            (3, 4),
        ]

        for a1, a2 in anatomy_topo_pairs:
            pos1 = anat_base + a1
            pos2 = anat_base + a2
            mask[pos1, pos2] = 0.0
            mask[pos2, pos1] = 0.0

        return mask

    def forward(self, nodes):
        P = self.num_patches
        B, N, C = nodes.shape

        patch_nodes = nodes[:, : 3 * P, :].view(B, 3, P, C)
        patch_nodes = patch_nodes + self.patch_pos_embedding
        patch_nodes = patch_nodes.view(B, 3 * P, C)

        anatomy_nodes = nodes[:, 3 * P :, :]
        anatomy_nodes = anatomy_nodes + self.anatomy_pos_embedding

        nodes = torch.cat([patch_nodes, anatomy_nodes], dim=1)

        return self.transformer(
            nodes,
            mask=self.attention_mask,
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


class ResUNetPPDecoder3D(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.d1 = DecoderBlock3D(gate_channels=256, skip_channels=64, out_channels=128)
        self.d2 = DecoderBlock3D(gate_channels=128, skip_channels=32, out_channels=64)
        self.d3 = DecoderBlock3D(gate_channels=64, skip_channels=32, out_channels=32)
        self.aspp = ASPP3D(32, 16)
        self.head = nn.Conv3d(16, num_classes, kernel_size=1)

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
        x = self.d1(bottleneck, skip3)
        if cond is not None:
            x = self.apply_film(x, self.film_d1, cond)

        x = self.d2(x, skip2)
        if cond is not None:
            x = self.apply_film(x, self.film_d2, cond)

        x = self.d3(x, skip1)
        if cond is not None:
            x = self.apply_film(x, self.film_d3, cond)

        x = self.aspp(x)
        return self.head(x)


class ResUNetPP3DMultiHead(nn.Module):
    def __init__(self, in_channels=1, source_order=("2ch", "4ch", "sa"), num_classes_by_source=None):
        super().__init__()
        self.source_order = list(source_order)
        self.num_classes_by_source = num_classes_by_source or {"2ch": 3, "4ch": 6, "sa": 4}

        self.patch_size = (2, 2, 2)
        self.patch_grid = (4, 10, 10)
        self.num_patches = self.patch_grid[0] * self.patch_grid[1] * self.patch_grid[2]

        self.encoder = nn.ModuleDict({
            "c1": StemBlock3D(in_channels, 32, 1),
            "c2": ResBlock3D(32, 32, 2),
            "c3": ResBlock3D(32, 64, 2),
            "c4": ResBlock3D(64, 128, 2),
            "b": ASPP3D(128, 256),
        })

        self.patch_proj = nn.Conv3d(
            in_channels=256,
            out_channels=256,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))

        # 模态标识嵌入向量: [1, 1, 256]
        self.modality_embedding = nn.ParameterDict({
            "2ch": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
            "4ch": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
            "sa": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
        })

        # 5 个解剖结构的 [CLS] Tokens (LV_myo, LV_cav, RV_cav, RA, LA)
        self.anatomy_tokens = nn.Parameter(torch.randn(5, 256) * 0.02)
        
        # 解剖掩码 Transformer 颈部
        self.transformer_neck = AnatomyMaskedTransformer(
            feature_dim=256,
            num_patches_per_view=self.num_patches,
            num_anatomy_nodes=5,
            heads=8,
            layers=2,
        )

        # 生理连通的解剖结构索引
        self.anatomy_indices_by_view = {
            "2ch": [0, 1],               # LV_myo, LV_cav
            "4ch": [0, 1, 2, 3, 4],      # 全部 5 个结构
            "sa": [0, 1, 2],            # LV_myo, LV_cav, RV_cav
        }

        # 将连通的解剖 CLS Token 聚合映射为当前切面的条件向量 cond [B, 256]
        self.anatomy_cond_proj = nn.ModuleDict({
            src: nn.Sequential(
                nn.Linear(len(self.anatomy_indices_by_view[src]) * 256, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
            )
            for src in self.source_order
        })

        # 允许 Transformer 从第 0 轮介入反向传播
        self.gamma_patch = nn.Parameter(torch.ones(1) * 0.1)

        self.decoders = nn.ModuleDict()
        for src in self.source_order:
            self.decoders[src] = ResUNetPPDecoder3D(self.num_classes_by_source[src])

    def encode(self, x, modality_emb=None):
        """
        共享 Encoder 编码逻辑：接收单切面输入 x，并将传入的 modality_embedding 共享注入
        x: [B, 1, D, H, W]
        modality_emb: [1, 1, 256] (或 [B, 1, 256])
        """
        s1 = self.encoder["c1"](x)
        s2 = self.encoder["c2"](s1)
        s3 = self.encoder["c3"](s2)
        s4 = self.encoder["c4"](s3)
        b = self.encoder["b"](s4)

        if modality_emb is not None:
            emb_bias = modality_emb.view(-1, 256, 1, 1, 1)
            b = b + emb_bias

        return s1, s2, s3, b

    def feature_to_token(self, b):
        x_proj = self.patch_proj(b)
        B, C, _, _, _ = x_proj.shape
        return x_proj.view(B, C, -1).transpose(1, 2)

    def forward(self, x, modality_mask):
        """x: [B,3,D,H,W], modality_mask: [B,3]"""
        B = x.shape[0]
        P = self.num_patches
        modality_features = []
        skips = {}

        for i, src in enumerate(self.source_order):
            present = modality_mask[:, i]
            xi = x[:, i : i + 1]
            mod_emb = self.modality_embedding[src]

            s1, s2, s3, b = self.encode(xi, modality_emb=mod_emb)
            skips[src] = (s1, s2, s3, b)

            token = self.feature_to_token(b)  # [B, P, 256]
            missing = (present == 0)

            if missing.any():
                token = token.clone()
                token[missing] = self.mask_token.expand(missing.sum(), P, -1)

            # 同时在 Token 层面保留模态属性偏置
            token = token + mod_emb
            modality_features.append(token)

        modality_nodes = torch.cat(modality_features, dim=1)
        anatomy_nodes = self.anatomy_tokens.unsqueeze(0).expand(B, -1, -1)

        # 联合序列送入带解剖拓扑掩码的 Transformer 交互网络
        tokens = torch.cat([modality_nodes, anatomy_nodes], dim=1)
        neck_out = self.transformer_neck(tokens)

        updated_patches = neck_out[:, : 3 * P, :].view(B, 3, P, 256)
        updated_anatomy = neck_out[:, 3 * P :, :]  # [B, 5, 256]

        outputs = {}
        for i, src in enumerate(self.source_order):
            s1, s2, s3, b = skips[src]
            target_shape = b.shape[2:]

            # 1. 局部 Patch 空间增强
            view_patches = updated_patches[:, i]
            patch_feat = view_patches.transpose(1, 2).view(B, 256, *self.patch_grid)
            patch_context_feat = F.interpolate(patch_feat, size=target_shape, mode="trilinear", align_corners=False)
            b = b + self.gamma_patch * patch_context_feat

            # 2. 提取当前视图对应连通的解剖 [CLS] Tokens 并映射为条件向量 cond
            sel_indices = self.anatomy_indices_by_view[src]
            sel_tokens = updated_anatomy[:, sel_indices, :]  # [B, K, 256]
            cond = self.anatomy_cond_proj[src](sel_tokens.reshape(B, -1))  # [B, 256]

            # 3. 传入多阶段 FiLM 解码器
            outputs[src] = self.decoders[src](s1, s2, s3, b, cond=cond)

        return {
            "logits": outputs,
        }