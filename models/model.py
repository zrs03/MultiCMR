from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyGraphTransformer(nn.Module):

    def __init__(self, feature_dim=256, num_nodes=8, heads=8, layers=2):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_nodes, feature_dim)
        )

        self.register_buffer(
            "graph_mask",
            self.build_graph_mask()
        )


    def build_graph_mask(self):

        A = torch.zeros(8, 8)


        # node definition:
        #
        # 0 : 2CH
        # 1 : 4CH
        # 2 : SA
        #
        # 3 : LV_myo
        # 4 : LV_cav
        # 5 : RV_cav
        # 6 : RA
        # 7 : LA


        edges = [

            # =====================
            # view-view
            # =====================
            (0,1),    # 2CH - 4CH
            (0,2),    # 2CH - SA
            (1,2),    # 4CH - SA


            # =====================
            # 2CH anatomy
            # =====================
            (0,3),    # 2CH - LV_myo
            (0,4),    # 2CH - LV_cav


            # =====================
            # 4CH anatomy
            # =====================
            (1,3),    # 4CH - LV_myo
            (1,4),    # 4CH - LV_cav
            (1,5),    # 4CH - RV_cav
            (1,6),    # 4CH - RA
            (1,7),    # 4CH - LA


            # =====================
            # SA anatomy
            # =====================
            (2,3),    # SA - LV_myo
            (2,4),    # SA - LV_cav
            (2,5),    # SA - RV_cav

        ]


        for i,j in edges:
            A[i,j]=1
            A[j,i]=1


        # self connection
        for i in range(8):
            A[i,i]=1


        mask=torch.where(
            A>0,
            torch.tensor(0.0),
            torch.tensor(float("-inf"))
        )


        return mask



    def forward(self,nodes):

        """
        nodes:
        [B,8,256]

        index:
        0 : 2CH
        1 : 4CH
        2 : SA

        3 : LV_myo
        4 : LV_cav
        5 : RV_cav
        6 : RA
        7 : LA
        """

        nodes = nodes + self.pos_embedding

        return self.transformer(
            nodes,
            mask=self.graph_mask
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
        self.encoder = nn.ModuleDict({
            "c1": StemBlock3D(1, 16, 1),
            "c2": ResBlock3D(16, 32, 2),
            "c3": ResBlock3D(32, 64, 2),
            "c4": ResBlock3D(64, 128, 2),
            "b": ASPP3D(128, 256),
        })
        self.missing_tokens = nn.ParameterDict({
            "2ch": nn.Parameter(torch.randn(256)),
            "4ch": nn.Parameter(torch.randn(256)),
            "sa": nn.Parameter(torch.randn(256)),
        })
        self.modality_embedding = nn.ParameterDict({
            "2ch": nn.Parameter(torch.randn(256)),
            "4ch": nn.Parameter(torch.randn(256)),
            "sa": nn.Parameter(torch.randn(256)),
        })
        self.anatomy_tokens = nn.Parameter(torch.randn(5, 256))
        self.graph = AnatomyGraphTransformer(feature_dim=256, num_nodes=8, heads=8, layers=2)
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
        """b: [B,256,D,H,W] -> [B,256]"""
        return F.adaptive_avg_pool3d(b, 1).flatten(1)

    def forward(self, x, modality_mask):
        """x: [B,3,D,H,W], modality_mask: [B,3]"""
        B = x.shape[0]
        modality_features = []
        skips = {}
        for i, src in enumerate(self.source_order):
            present = modality_mask[:, i]
            xi = x[:, i:i+1]
            s1, s2, s3, b = self.encode(xi)
            skips[src] = (s1, s2, s3, b)
            token = self.feature_to_token(b)
            missing = (present == 0)
            if missing.any():
                token = token.clone()
                token[missing] = self.missing_tokens[src].unsqueeze(0).expand(missing.sum(), -1)
            token = token + self.modality_embedding[src]
            modality_features.append(token)
        modality_nodes = torch.stack(modality_features, dim=1)
        anatomy_nodes = self.anatomy_tokens.unsqueeze(0).expand(B, -1, -1)
        nodes = torch.cat([modality_nodes, anatomy_nodes], dim=1)
        graph_out = self.graph(nodes)
        outputs = {}
        for i, src in enumerate(self.source_order):
            s1, s2, s3, b = skips[src]
            graph_token = graph_out[:, i, :, None, None, None]
            graph_token = graph_token.expand(-1, -1, b.shape[2], b.shape[3], b.shape[4])
            b = b + graph_token
            outputs[src] = self.decoders[src](s1, s2, s3, b)
        return {"logits": outputs, "graph_nodes": graph_out}
