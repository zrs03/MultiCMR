from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyGraphTransformer(nn.Module):
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
        self.heads = heads
        self.total_nodes = 3 * num_patches_per_view + num_anatomy_nodes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
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

        # 静态解剖先验掩码（基底拓扑）
        self.register_buffer(
            "base_graph_mask",
            self.build_graph_mask(),
        )

    def build_graph_mask(self):
        P = self.num_patches
        N = self.total_nodes
        A = torch.zeros(N, N)

        # 1. 视图内部 Patch 自连
        for v in range(3):
            A[v * P : (v + 1) * P, v * P : (v + 1) * P] = 1.0

        # 2. 跨视图间 Patch 互通
        view_view_pairs = [(0, 1), (0, 2), (1, 2)]
        for v1, v2 in view_view_pairs:
            A[v1 * P : (v1 + 1) * P, v2 * P : (v2 + 1) * P] = 1.0
            A[v2 * P : (v2 + 1) * P, v1 * P : (v1 + 1) * P] = 1.0

        # 3. 视图 Patch 与特定解剖先验 Token 交互
        anat_offset = 3 * P
        anatomy_view_mapping = [
            (0, [0, 1, 2]),  # LV_myo <-> 2CH, 4CH, SA
            (1, [0, 1, 2]),  # LV_cav <-> 2CH, 4CH, SA
            (2, [1, 2]),     # RV_cav <-> 4CH, SA
            (3, [1]),        # RA     <-> 4CH
            (4, [1]),        # LA     <-> 4CH
        ]

        for anat_idx, views in anatomy_view_mapping:
            a_idx = anat_offset + anat_idx
            for v in views:
                A[v * P : (v + 1) * P, a_idx] = 1.0
                A[a_idx, v * P : (v + 1) * P] = 1.0

        # 4. 解剖先验 Token 之间互通
        A[anat_offset:, anat_offset:] = 1.0

        # 5. 自连接
        for i in range(N):
            A[i, i] = 1.0

        mask = torch.where(
            A > 0,
            torch.tensor(0.0),
            torch.tensor(float("-inf")),
        )
        return mask

    def build_dynamic_mask(self, modality_mask: torch.Tensor) -> torch.Tensor:
        """
        方案一核心：动态构造单向注意力掩码
        modality_mask: [B, 3]，1 表示存在，0 表示缺失
        输出: [B * heads, N, N]
        """
        B = modality_mask.shape[0]
        P = self.num_patches
        N = self.total_nodes

        # 复制基础解剖拓扑掩码: [B, N, N]
        mask = self.base_graph_mask.unsqueeze(0).repeat(B, 1, 1).clone()

        # 对每个缺失的视图，阻断其它节点看它的视线（作为 Key/Value 被屏蔽）
        for v in range(3):
            # missing_idx: 批次中缺失视图 v 的样本索引
            missing = (modality_mask[:, v] == 0)
            if missing.any():
                v_start = v * P
                v_end = (v + 1) * P
                
                # 关键：当视图 v 缺失时，任何其他健康节点和解剖节点，
                # 都不能将注意力放在视图 v 上（mask[..., :, v_start:v_end] 置为 -inf）
                mask[missing, :, v_start:v_end] = float("-inf")
                
                # 保留自环（防止全为 -inf 出现 NaN）
                for p_idx in range(v_start, v_end):
                    mask[missing, p_idx, p_idx] = 0.0

        # 适配 multi-head 维度: [B * heads, N, N]
        mask = mask.repeat_interleave(self.heads, dim=0)
        return mask

    def forward(self, nodes, modality_mask=None):
        """
        nodes: [B, 3 * P + 5, 256]
        modality_mask: [B, 3]
        """
        P = self.num_patches
        B, N, C = nodes.shape

        patch_nodes = nodes[:, : 3 * P, :].view(B, 3, P, C)
        patch_nodes = patch_nodes + self.patch_pos_embedding
        patch_nodes = patch_nodes.view(B, 3 * P, C)

        anatomy_nodes = nodes[:, 3 * P :, :]
        anatomy_nodes = anatomy_nodes + self.anatomy_pos_embedding

        nodes = torch.cat([patch_nodes, anatomy_nodes], dim=1)

        # 构造动态样本掩码
        if modality_mask is not None:
            dyn_mask = self.build_dynamic_mask(modality_mask)
        else:
            dyn_mask = self.base_graph_mask

        return self.transformer(
            nodes,
            mask=dyn_mask,
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
        self.d3 = DecoderBlock3D(gate_channels=64, skip_channels=16, out_channels=32)
        self.aspp = ASPP3D(32, 16)
        self.head = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, skip1, skip2, skip3, bottleneck):
        x = self.d1(bottleneck, skip3)
        x = self.d2(x, skip2)
        x = self.d3(x, skip1)
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
            "c1": StemBlock3D(in_channels, 16, 1),
            "c2": ResBlock3D(16, 32, 2),
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

        # 方案一：采用通用的可学习 mask token 代替每个模态独立的硬常数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))

        self.modality_embedding = nn.ParameterDict({
            "2ch": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
            "4ch": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
            "sa": nn.Parameter(torch.randn(1, 1, 256) * 0.02),
        })

        self.anatomy_tokens = nn.Parameter(torch.randn(5, 256) * 0.02)
        self.graph = AnatomyGraphTransformer(
            feature_dim=256,
            num_patches_per_view=self.num_patches,
            num_anatomy_nodes=5,
            heads=8,
            layers=2,
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.decoders = nn.ModuleDict()
        for src in self.source_order:
            self.decoders[src] = ResUNetPPDecoder3D(self.num_classes_by_source[src])

    def encode(self, x):
        s1 = self.encoder["c1"](x)
        s2 = self.encoder["c2"](s1)
        s3 = self.encoder["c3"](s2)
        s4 = self.encoder["c4"](s3)
        b = self.encoder["b"](s4)
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
            s1, s2, s3, b = self.encode(xi)
            skips[src] = (s1, s2, s3, b)

            token = self.feature_to_token(b)  # [B, P, 256]
            missing = (present == 0)

            # 方案一：缺失位置置为纯净的统一 mask_token，不再引入模态偏差
            if missing.any():
                token = token.clone()
                token[missing] = self.mask_token.expand(missing.sum(), P, -1)

            # 叠加模态辨识标识
            token = token + self.modality_embedding[src]
            modality_features.append(token)

        modality_nodes = torch.cat(modality_features, dim=1)
        anatomy_nodes = self.anatomy_tokens.unsqueeze(0).expand(B, -1, -1)

        nodes = torch.cat([modality_nodes, anatomy_nodes], dim=1)
        
        # 传入 modality_mask，图网络内部动态单向遮蔽缺失模态作为 Key 的注意力
        graph_out = self.graph(nodes, modality_mask=modality_mask)

        updated_patches = graph_out[:, : 3 * P, :].view(B, 3, P, 256)

        outputs = {}
        for i, src in enumerate(self.source_order):
            s1, s2, s3, b = skips[src]
            target_shape = b.shape[2:]

            view_patches = updated_patches[:, i]
            patch_feat = view_patches.transpose(1, 2).view(B, 256, *self.patch_grid)
            graph_token = F.interpolate(patch_feat, size=target_shape, mode="trilinear", align_corners=False)

            b = b + self.gamma * graph_token
            outputs[src] = self.decoders[src](s1, s2, s3, b)

        return {"logits": outputs, "graph_nodes": graph_out}