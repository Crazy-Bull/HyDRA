import math
from dataclasses import dataclass
from typing import Literal, Union

import torch
from torch import nn
from torch.nn import functional as F


class ResidualTemporalConv1d(nn.Module):
    """Residual multi-scale 1D convolution used by the CFRE block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 3,
        fixed_kernel_size: int = 15,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        temporal_padding = (temporal_kernel_size // 2) * dilation
        fixed_padding = fixed_kernel_size // 2

        self.temporal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=temporal_kernel_size,
            padding=temporal_padding,
            dilation=dilation,
        )
        self.fixed_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=fixed_kernel_size,
            padding=fixed_padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        features = self.temporal_conv(x) + self.fixed_conv(x)
        features = self.activation(self.norm(features))
        return features + identity


class ConvolutionalFeatureRefinementExtractor(nn.Module):
    """CFRE maps an IQ/VMD sequence from channel space to feature space."""

    def __init__(
        self,
        input_channels: int,
        feature_dim: int = 64,
        temporal_kernel_size: int = 3,
        fixed_kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ResidualTemporalConv1d(
                input_channels,
                32,
                temporal_kernel_size=temporal_kernel_size,
                fixed_kernel_size=fixed_kernel_size,
            ),
            ResidualTemporalConv1d(
                32,
                32,
                temporal_kernel_size=temporal_kernel_size,
                fixed_kernel_size=fixed_kernel_size,
                dilation=3,
            ),
            ResidualTemporalConv1d(
                32,
                feature_dim,
                temporal_kernel_size=temporal_kernel_size,
                fixed_kernel_size=fixed_kernel_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input with shape (batch, length, channels).")
        x = x.transpose(1, 2)
        x = self.layers(x)
        return x.transpose(1, 2)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, feature_dim: int, max_len: int = 257) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, feature_dim))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_embedding[:, : x.size(1), :]


class TransformerDynamicSequenceEncoder(nn.Module):
    """TDSE: Transformer encoder with a learnable class token."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 257,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.position = LearnablePositionalEncoding(feature_dim, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position(x)
        return self.encoder(x)

    def pool(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded[:, 0, :]


@dataclass
class MambaBlockArgs:
    feature_dim: int
    state_dim: int = 16
    conv_kernel_size: int = 4
    expansion_factor: int = 2
    dt_rank: Union[int, Literal["auto"]] = "auto"
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.feature_dim / 16)
        self.inner_dim = int(self.expansion_factor * self.feature_dim)


class MinimalMambaBlock(nn.Module):
    """Minimal selective SSM block used for the MLFE path."""

    def __init__(self, args: MambaBlockArgs) -> None:
        super().__init__()
        self.args = args
        inner_dim = args.inner_dim

        self.in_proj = nn.Linear(args.feature_dim, inner_dim * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=args.conv_kernel_size,
            groups=inner_dim,
            padding=args.conv_kernel_size - 1,
            bias=args.conv_bias,
        )
        self.x_proj = nn.Linear(
            inner_dim,
            int(args.dt_rank) + args.state_dim * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(int(args.dt_rank), inner_dim, bias=True)

        base = torch.arange(1, args.state_dim + 1, dtype=torch.float32)
        base = base.repeat(inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(base))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, args.feature_dim, bias=args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x_and_residual = self.in_proj(x)
        x, residual = x_and_residual.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = F.silu(x.transpose(1, 2))

        y = self._selective_ssm(x)
        y = y * F.silu(residual)
        return self.out_proj(y)

    def _selective_ssm(self, x: torch.Tensor) -> torch.Tensor:
        inner_dim, state_dim = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split(
            [int(self.args.dt_rank), state_dim, state_dim],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))
        return self._selective_scan(x, delta, A, B, C, D)

    @staticmethod
    def _selective_scan(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, inner_dim = u.shape
        state_dim = A.shape[1]

        delta_A = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        delta_B_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        state = torch.zeros(
            batch_size,
            inner_dim,
            state_dim,
            dtype=u.dtype,
            device=u.device,
        )
        outputs = []
        for t in range(seq_len):
            state = delta_A[:, t] * state + delta_B_u[:, t]
            outputs.append(torch.einsum("bdn,bn->bd", state, C[:, t, :]))

        y = torch.stack(outputs, dim=1)
        return y + u * D


class MambaLinearFlowEncoder(nn.Module):
    """MLFE: one or more minimal Mamba-style sequence blocks."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_layers: int = 1,
        state_dim: int = 16,
        conv_kernel_size: int = 4,
        expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        args = MambaBlockArgs(
            feature_dim=feature_dim,
            state_dim=state_dim,
            conv_kernel_size=conv_kernel_size,
            expansion_factor=expansion_factor,
        )
        self.layers = nn.ModuleList([MinimalMambaBlock(args) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def pool(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded.mean(dim=1)


class HyDRA(nn.Module):
    """Hybrid Dual-mode RF Architecture for RFFI."""

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 28,
        mode: Literal["tdse", "mlfe"] = "tdse",
        input_length: int = 256,
        feature_dim: int = 64,
        temporal_kernel_size: int = 3,
        fixed_kernel_size: int = 15,
        tdse_layers: int = 2,
        attention_heads: int = 4,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        mlfe_layers: int = 1,
        state_dim: int = 16,
        mamba_conv_kernel_size: int = 4,
        expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        if mode not in {"tdse", "mlfe"}:
            raise ValueError("mode must be either 'tdse' or 'mlfe'.")

        self.mode = mode
        self.cfre = ConvolutionalFeatureRefinementExtractor(
            input_channels=input_channels,
            feature_dim=feature_dim,
            temporal_kernel_size=temporal_kernel_size,
            fixed_kernel_size=fixed_kernel_size,
        )
        if mode == "tdse":
            self.encoder = TransformerDynamicSequenceEncoder(
                feature_dim=feature_dim,
                num_heads=attention_heads,
                num_layers=tdse_layers,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                max_len=input_length + 1,
            )
        else:
            self.encoder = MambaLinearFlowEncoder(
                feature_dim=feature_dim,
                num_layers=mlfe_layers,
                state_dim=state_dim,
                conv_kernel_size=mamba_conv_kernel_size,
                expansion_factor=expansion_factor,
            )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cfre(x)
        encoded = self.encoder(features)
        return self.encoder.pool(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.classifier(pooled)
