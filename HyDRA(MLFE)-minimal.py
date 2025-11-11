import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

# --- CNN Feature Extractor ---

class ResConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResConv1d, self).__init__()
        padding_t = (kernel_size // 2) * dilation
        padding_f = 7  # Fixed for kernel_size=15
        self.conv_t = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_t, dilation=dilation)
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding_f)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out_t = self.conv_t(x)
        out_f = self.conv_f(x)
        out = out_t + out_f
        out = self.bn(out)
        out = self.relu(out)
        return out + identity

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=2, d_model=64):
        super(CNNFeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            ResConv1d(input_channels, 32),
            ResConv1d(32, 32, dilation=3),
            ResConv1d(32, d_model)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # e.g., (batch_size, 2, 256)
        x = self.layers(x)  # (batch_size, d_model, 256)
        return x.permute(0, 2, 1)  # (batch_size, 256, d_model)

# --- Mamba-minimal Components ---

class ModelArgs:
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto', conv_bias=True, bias=False):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank if dt_rank != 'auto' else math.ceil(d_model / 16)
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_inner = int(self.expand * self.d_model)

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block from Mamba-minimal."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # (b, l, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (b, l, d_in)
        y = y + u * D
        return y

# --- Updated Single-Layer Mamba Encoder ---

class MambaEncoder(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        args = ModelArgs(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.layer = MambaBlock(args)

    def forward(self, x):
        x = self.layer(x)  # Single Mamba layer
        return x

# --- CNN+Mamba Model with Single-Layer Mamba ---

class CNNMambaModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=10, d_model=64, 
                 d_state=16, d_conv=4, expand=2):
        super(CNNMambaModel, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, d_model)
        self.mamba_encoder = MambaEncoder(d_model, d_state, d_conv, expand)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn(x)          # (batch_size, 256, d_model)
        x = self.mamba_encoder(x) # (batch_size, 256, d_model)
        x = x.mean(dim=1)        # (batch_size, d_model)
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits
    
# --- Training and Evaluation ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_channels = k  # k VMD components per signal sample
num_classes = len(tx_list)  # Number of transmitters

# Initialize model, loss, and optimizer
model = CNNMambaModel(input_channels=input_channels, num_classes=num_classes).to(device)