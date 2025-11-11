import torch
import torch.nn as nn
from mamba_ssm import Mamba

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
    def __init__(self, input_channels=2, d_model=64, split_channels=None):
        super(CNNFeatureExtractor, self).__init__()
        if split_channels is not None:
            assert input_channels == sum(split_channels), "Input channels must match the sum of split channels"
            self.split = True
            self.split_channels = split_channels
            # Magnitude path
            self.mag_cnn = nn.Sequential(
                ResConv1d(split_channels[0], 32),
                ResConv1d(32, 64)
            )
            # Phase path
            self.phase_cnn = nn.Sequential(
                ResConv1d(split_channels[1], 32),
                ResConv1d(32, 64)
            )
            # Combine magnitude and phase features
            self.combine = nn.Conv1d(128, d_model, kernel_size=1)  # 64 (mag) + 64 (phase) = 128
        else:
            self.split = False
            self.layers = nn.Sequential(
                ResConv1d(input_channels, 32),
                ResConv1d(32, 32, dilation=3),
                ResConv1d(32, d_model)
            )

    def forward(self, x):
        if self.split:
            mag = x[:, :, :self.split_channels[0]]  # e.g., (batch_size, 256, 5)
            phase = x[:, :, self.split_channels[0]:]  # e.g., (batch_size, 256, 5)
            mag = mag.permute(0, 2, 1)  # (batch_size, 5, 256)
            phase = phase.permute(0, 2, 1)  # (batch_size, 5, 256)
            mag_feat = self.mag_cnn(mag)  # (batch_size, 64, 256)
            phase_feat = self.phase_cnn(phase)  # (batch_size, 64, 256)
            combined = torch.cat((mag_feat, phase_feat), dim=1)  # (batch_size, 128, 256)
            combined = self.combine(combined)  # (batch_size, d_model, 256)
            return combined.permute(0, 2, 1)  # (batch_size, 256, d_model)
        else:
            x = x.permute(0, 2, 1)  # e.g., (batch_size, 2, 256)
            x = self.layers(x)  # (batch_size, d_model, 256)
            return x.permute(0, 2, 1)  # (batch_size, 256, d_model)

# --- Multi-Layer Mamba Encoder ---

class MambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers=2, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Pure Mamba layers, no residuals or normalization
        return x

# --- CNN+Mamba Model with Multi-Layer Mamba ---

class CNNMambaModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=10, d_model=64, num_layers=2, 
                 d_state=16, d_conv=4, expand=2, split_channels=None):
        super(CNNMambaModel, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, d_model, split_channels)
        self.mamba_encoder = MambaEncoder(d_model, num_layers, d_state, d_conv, expand)
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
model = CNNMambaModel(input_channels=input_channels, num_classes=num_classes, split_channels=None).to(device)
# model = CNNMambaModel(input_channels=input_channels, num_classes=num_classes, split_channels=(5, 5)).to(device)