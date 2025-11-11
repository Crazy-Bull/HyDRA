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
tx_list = ['1-11', '10-11', '10-7', '11-1', '11-17', '11-4', '11-7', '13-3', '14-10', '14-7', '15-1', '16-16', '2-19', '20-12', '20-15', '20-19', '20-7', '3-13', '3-18', '4-11', '5-5', '6-1', '6-15', '7-10', '7-11', '8-18', '8-20', '8-3']  # List of transmitter names
rx_list = ['1-1', '13-13', '14-7', '2-1', '2-20', '20-1', '7-14', '7-7', '8-13', '8-8']                        # List of receiver names
capture_date_list = ['2021_03_23'] # List of capture dates
dataset_path = 'D:/Material/UCLA SCI/'
name = 'amp_phase_5.pkl'
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

# --- CNN+Transformer Model ---

class ResConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResConv1d, self).__init__()
        padding = (kernel_size // 2) * dilation  # 'same' padding to keep seq_len=256
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + identity

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=2, d_model=64):
        super(CNNFeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            ResConv1d(input_channels, 32, kernel_size=3, dilation=1),
            ResConv1d(32, 32, kernel_size=3, dilation=3),  # Multi-scale with dilation
            ResConv1d(32, d_model, kernel_size=3, dilation=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, 2, 256)
        x = self.layers(x)      # (batch_size, d_model, 256)
        x = x.permute(0, 2, 1)  # (batch_size, 256, d_model)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=257):  # 256 + cls_token
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
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, 0, 0.02)

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)                  # (batch_size, 257, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)                                # (batch_size, 257, d_model)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, d_model=64, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        logits = self.fc(cls_output)  # (batch_size, num_classes)
        return logits

class CNNTransformerModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(CNNTransformerModel, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, d_model)
        self.transformer = TransformerEncoderModel(d_model, nhead, num_layers, dim_feedforward)
        self.classifier = ClassificationHead(d_model, num_classes)

    def forward(self, x):
        x = self.cnn(x)         # (batch_size, 256, d_model)
        x = self.transformer(x) # (batch_size, 257, d_model)
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits

# --- Training and Evaluation ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_dim = k  # k VMD components per signal sample
num_classes = len(tx_list)  # Number of transmitters

# Initialize model, loss, and optimizer
model = CNNTransformerModel(input_dim, num_classes).to(device)
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