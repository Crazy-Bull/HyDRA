class ResConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResConv1d, self).__init__()
        padding_t = (kernel_size // 2) * dilation
        padding_f = 7  # Fixed for kernel_size=15
        self.pad_t = nn.ReflectionPad1d(padding_t)
        self.pad_f = nn.ReflectionPad1d(padding_f)
        self.conv_t = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size=15)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)  # Updated from ReLU
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out_t = self.conv_t(self.pad_t(x))
        out_f = self.conv_f(self.pad_f(x))
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
            self.mag_cnn = nn.Sequential(
                ResConv1d(split_channels[0], 32, dilation=1),
                ResConv1d(32, 64, dilation=3)
            )
            self.phase_cnn = nn.Sequential(
                ResConv1d(split_channels[1], 32, dilation=1),
                ResConv1d(32, 64, dilation=3)
            )
            self.combine = nn.Conv1d(128, d_model, kernel_size=1)
        else:
            self.split = False
            self.layers = nn.Sequential(
                ResConv1d(input_channels, 32, dilation=1),
                ResConv1d(32, 32, dilation=3),
                ResConv1d(32, d_model, dilation=9)
            )

    def forward(self, x):
        if self.split:
            mag = x[:, :, :self.split_channels[0]]
            phase = x[:, :, self.split_channels[0]:]
            mag = mag.permute(0, 2, 1)
            phase = phase.permute(0, 2, 1)
            mag_feat = self.mag_cnn(mag)
            phase_feat = self.phase_cnn(phase)
            combined = torch.cat((mag_feat, phase_feat), dim=1)
            combined = self.combine(combined)
            return combined.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.layers(x)
            return x.permute(0, 2, 1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=257):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, 0, 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, 0, 0.02)

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, d_model=64, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model // 2, num_classes)

    def forward(self, x):
        cls_output = x[:, 0, :]
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class CNNTransformerModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=10, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, split_channels=None):
        super(CNNTransformerModel, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, d_model, split_channels)
        self.transformer = TransformerEncoderModel(d_model, nhead, num_layers, dim_feedforward)
        self.classifier = ClassificationHead(d_model, num_classes)
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
        x = self.cnn(x)        # (batch_size, 256, d_model)
        x = self.transformer(x) # (batch_size, 257, d_model)
        logits = self.classifier(x) # (batch_size, num_classes)
        return logits

# --- Training and Evaluation ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_channels = k  # k VMD components per signal sample
num_classes = len(tx_list)  # Number of transmitters

# Initialize model, loss, and optimizer
model = CNNTransformerModel(input_channels=input_channels, num_classes=num_classes, split_channels=None).to(device)
# model = CNNTransformerModel(input_channels=input_channels, num_classes=num_classes, split_channels=(5, 5)).to(device)