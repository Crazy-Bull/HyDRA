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

# --- Single-Layer Mamba Encoder ---

class MambaEncoder(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        self.layer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

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