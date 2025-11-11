import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pickle
import os
import os.path
from sktime.libs.vmdpy import VMD
from pathlib import Path


def load_processed_data(dataset_path, name):
    """加载已处理的数据"""
    SAVE_DIR = Path(f'{dataset_path}/data_prep')
    try:
        with open(SAVE_DIR/name, 'rb') as f:
            data = pickle.load(f)
        print("Loaded preprocessed data")
        return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['class_weights'], data['params']
    except FileNotFoundError:
        print("No preprocessed data found")
        return None, None, None, None, None, None, None, None
    
# Define parameters (replace with actual values from your dataset)
tx_list = ['1-1', '1-10', '1-11', '1-12', '1-14', '1-15', '1-16', '1-18', '1-19', '1-2', '1-8', '10-1', '10-10', '10-11', '10-17', '10-4', '10-7', '11-1', '11-10', '11-17', '11-19', '11-20', '11-4', '11-7', '12-1', '12-19', '12-20', '12-7', '13-14', '13-18', '13-19', '13-20', '13-3', '13-7', '14-10', '14-11', '14-12', '14-13', '14-14', '14-20', '14-7', '14-8', '14-9', '15-1', '15-19', '15-6', '16-1', '16-16', '16-19', '16-20', '16-5', '17-10', '17-11', '18-1', '18-10', '18-11', '18-12', '18-13', '18-14', '18-15', '18-16', '18-17', '18-2', '18-20', '18-4', '18-5', '18-7', '18-8', '18-9', '19-1', '19-10', '19-11', '19-12', '19-13', '19-14', '19-19', '19-2', '19-20', '19-3', '19-4', '19-6', '19-7', '19-8', '19-9', '2-1', '2-12', '2-13', '2-14', '2-15', '2-16', '2-17', '2-19', '2-20', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '20-1', '20-12', '20-14', '20-15', '20-16', '20-18', '20-19', '20-20', '20-3', '20-4', '20-5', '20-7', '20-8', '3-1', '3-13', '3-18', '3-19', '3-2', '3-20', '3-8', '4-1', '4-10', '4-11', '5-1', '5-16', '5-20', '5-5', '6-1', '6-15', '6-6', '7-10', '7-11', '7-12', '7-13', '7-14', '7-20', '7-7', '7-8', '7-9', '8-1', '8-13', '8-14', '8-18', '8-20', '8-3', '8-7', '8-8', '9-1', '9-14', '9-20', '9-7']                       # List of receiver names
rx_list=['1-1', '1-19', '1-20', '13-7', '14-7', '18-19', '18-2', '19-1', '19-2', '2-1', '20-1', '20-19', '3-19', '7-14', '7-7', '8-14', '8-7', '8-8']
capture_date_list = ['2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23'] # List of capture dates
dataset_path = '/home/qiu/桌面/RF-fingerprint-classification/'
name = 'amp_phase_5_ManyTx.pkl'
X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params = load_processed_data(dataset_path, name)
k = params['k']

print("X_train shape:", X_train.shape)
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Now X_train, y_train, etc., are ready for model training

# --- Helper Classes ---

class ResBlock1dTF(nn.Module):
    """Residual block with time-frequency processing for 1D signals."""
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        # Temporal block with dilated convolution
        self.block_t = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size // 2)),  # Padding to maintain sequence length
            nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, bias=False, dilation=dilation, groups=dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        # Frequency block with 1x1 convolution
        self.block_f = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        # Shortcut connection
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)

    def forward(self, x):
        # Combine shortcut, frequency, and temporal paths
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)

class ClassificationHead(nn.Module):
    """Classification head that processes the CLS token output from the Transformer."""
    def __init__(self, input_dim=512, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Linear layer for final classification

    def forward(self, x):
        # Input shape: (257, batch_size, embed_dim)
        x = x[0]  # Extract CLS token output: (batch_size, embed_dim)
        x = self.fc(x)  # Output: (batch_size, num_classes)
        return x

class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=512, n_layers=6, nhead=6, dim_feedforward=512):
        super(TAggregate, self).__init__()
        self.num_tokens = 1
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, activation="gelu",
            dim_feedforward=dim_feedforward, dropout=drop_rate
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        # Corrected CLS token shape
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, embed_dim)
        # Positional embedding for sequence length + CLS token
        self.pos_embed = nn.Parameter(torch.zeros(256 + 1, 1, embed_dim))  # (257, 1, embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            with torch.no_grad():
                m.data.normal_(0.0, 0.02)

    def forward(self, x):
        # Input x: (256, batch_size, embed_dim)
        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(-1, x.shape[1], -1)  # (1, batch_size, embed_dim)
        # Concatenate along sequence dimension
        x = torch.cat((cls_tokens, x), dim=0)  # (257, batch_size, embed_dim)
        # Add positional embedding
        x = x + self.pos_embed  # (257, batch_size, embed_dim)
        # Process through Transformer
        x = self.transformer_enc(x)  # (257, batch_size, embed_dim)
        return x

# --- Main Model Class ---

class ModifiedModel(nn.Module):
    """Modified model for RFFI with dual CNN branches and Transformer."""
    def __init__(self, nf=32, embed_dim=128, n_layers=4, nhead=8, n_classes=None, dim_feedforward=512):
        super().__init__()

        # Magnitude branch: Processes 5 magnitude channels
        self.mag_branch = nn.Sequential(
            nn.Conv1d(5, nf, kernel_size=3, padding=1, bias=False),  # Output: (batch_size, nf, 256)
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(nf, nf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True)
        )

        # Phase branch: Processes 5 phase channels
        self.phase_branch = nn.Sequential(
            nn.Conv1d(5, nf, kernel_size=3, padding=1, bias=False),  # Output: (batch_size, nf, 256)
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(nf, nf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True)
        )

        # Fusion layer: Combines magnitude and phase features
        self.fusion = nn.Conv1d(nf * 2, nf, kernel_size=1, bias=False)  # Input: nf*2, Output: nf
        self.fusion_bn = nn.BatchNorm1d(nf)
        self.fusion_act = nn.LeakyReLU(0.2, True)

        # Feature enhancement with residual blocks (no downsampling)
        self.feature_enhance = nn.Sequential(
            ResBlock1dTF(dim=nf, dilation=1, kernel_size=15),
            ResBlock1dTF(dim=nf, dilation=3, kernel_size=15)
        )

        # Projection to Transformer embedding dimension
        self.project = nn.Conv1d(nf, embed_dim, kernel_size=1, bias=False)

        # Transformer for sequence processing
        self.tf = TAggregate(
            clip_length=256,  # Fixed sequence length
            embed_dim=embed_dim,
            n_layers=n_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )

        # Classification head for final output
        self.outputHead = ClassificationHead(input_dim=embed_dim, num_classes=n_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for convolutional and batch norm layers."""
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input shape: (batch_size, 256, 10)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, 10, 256) for convolution

        # Split into magnitude and phase components
        mag = x[:, :5, :]   # (batch_size, 5, 256)
        phase = x[:, 5:, :] # (batch_size, 5, 256)

        # Process through dual CNN branches
        mag_feat = self.mag_branch(mag)      # (batch_size, nf, 256)
        phase_feat = self.phase_branch(phase)  # (batch_size, nf, 256)

        # Concatenate and fuse features
        combined = torch.cat([mag_feat, phase_feat], dim=1)  # (batch_size, nf*2, 256)
        fused = self.fusion(combined)  # (batch_size, nf, 256)
        fused = self.fusion_bn(fused)
        fused = self.fusion_act(fused)

        # Enhance features without downsampling
        x = self.feature_enhance(fused)  # (batch_size, nf, 256)

        # Project to embedding dimension
        x = self.project(x)  # (batch_size, embed_dim, 256)

        # Reshape for Transformer
        x = x.permute(2, 0, 1)  # (256, batch_size, embed_dim)

        # Process through Transformer
        x = self.tf(x)  # (257, batch_size, embed_dim)

        # Classify using the CLS token output
        pred = self.outputHead(x)  # (batch_size, n_classes)
        return pred

# --- Training and Evaluation ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_dim = k  # k VMD components per signal sample
num_classes = len(tx_list)  # Number of transmitters

# Initialize model, loss, and optimizer
model = ModifiedModel(nf=32, embed_dim=128, n_layers=4, nhead=8, n_classes=num_classes, dim_feedforward=512).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.argmax(dim=1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))
            val_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Testing
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.argmax(dim=1).numpy())

# Convert predictions and true labels to numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=tx_list))