import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import pickle
import os
import os.path
from sktime.libs.vmdpy import VMD
from pathlib import Path


# --- init part ---

# Define parameters (replace with actual values from your dataset)
tx_list = ['1-1', '1-10', '1-11', '1-12', '1-14', '1-15', '1-16', '1-18', '1-19', '1-2', '1-8', '10-1', '10-10', '10-11', '10-17', '10-4', '10-7', '11-1', '11-10', '11-17', '11-19', '11-20', '11-4', '11-7', '12-1', '12-19', '12-20', '12-7', '13-14', '13-18', '13-19', '13-20', '13-3', '13-7', '14-10', '14-11', '14-12', '14-13', '14-14', '14-20', '14-7', '14-8', '14-9', '15-1', '15-19', '15-6', '16-1', '16-16', '16-19', '16-20', '16-5', '17-10', '17-11', '18-1', '18-10', '18-11', '18-12', '18-13', '18-14', '18-15', '18-16', '18-17', '18-2', '18-20', '18-4', '18-5', '18-7', '18-8', '18-9', '19-1', '19-10', '19-11', '19-12', '19-13', '19-14', '19-19', '19-2', '19-20', '19-3', '19-4', '19-6', '19-7', '19-8', '19-9', '2-1', '2-12', '2-13', '2-14', '2-15', '2-16', '2-17', '2-19', '2-20', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '20-1', '20-12', '20-14', '20-15', '20-16', '20-18', '20-19', '20-20', '20-3', '20-4', '20-5', '20-7', '20-8', '3-1', '3-13', '3-18', '3-19', '3-2', '3-20', '3-8', '4-1', '4-10', '4-11', '5-1', '5-16', '5-20', '5-5', '6-1', '6-15', '6-6', '7-10', '7-11', '7-12', '7-13', '7-14', '7-20', '7-7', '7-8', '7-9', '8-1', '8-13', '8-14', '8-18', '8-20', '8-3', '8-7', '8-8', '9-1', '9-14', '9-20', '9-7'] # List of transmitter names
rx_list = ['1-1', '13-13', '14-7', '2-1', '2-20', '20-1', '7-14', '7-7', '8-13', '8-8']                        # List of receiver names
capture_date_list = ['2021_03_23'] # List of capture dates
dataset_path = 'D:/git_workspace/RF-fingerprint-classification/ManyTx'
name = 'lossless_iq_2.pkl'


def load_processed_data(dataset_path, name, kfold_id = 0):
    """加载已处理的数据"""
    SAVE_DIR = Path(f'{dataset_path}/data_prep')
    try:
        with open(SAVE_DIR/name, 'rb') as f:
            data = pickle.load(f)
        print("Loaded preprocessed data")

        if kfold_id != 0:
            test_len = data['y_test'].shape[0]
            val_len = data['y_val'].shape[0]
            start_exchange_id = (test_len + val_len) * (kfold_id-1)
            tmp = data['X_train'][start_exchange_id:start_exchange_id+val_len]
            data['X_train'][start_exchange_id:start_exchange_id+val_len] = data['X_val']
            data['X_val'] = tmp
            tmp = data['y_train'][start_exchange_id:start_exchange_id+val_len]
            data['y_train'][start_exchange_id:start_exchange_id+val_len] = data['y_val']
            data['y_val'] = tmp
            tmp = data['X_train'][start_exchange_id+val_len:start_exchange_id+val_len+test_len]
            data['X_train'][start_exchange_id+val_len:start_exchange_id+val_len+test_len] = data['X_test']
            data['X_test'] = tmp
            tmp = data['y_train'][start_exchange_id+val_len:start_exchange_id+val_len+test_len]
            data['y_train'][start_exchange_id+val_len:start_exchange_id+val_len+test_len] = data['y_test']
            data['y_test'] = tmp
            
        return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['class_weights'], data['params']
    except FileNotFoundError:
        print("No preprocessed data found")
        return None, None, None, None, None, None, None, None



# --- define your model here ---

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
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 257, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch_size, 257, d_model)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, d_model=64, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        return self.fc(cls_output)  # (batch_size, num_classes)

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
def train_eval(k_fold_id = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params = load_processed_data(dataset_path, name, k_fold_id)
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

    # --- general training config ---

    num_epochs = 1000   # maximum epochs
    enable_lr_reduction = True
    min_lr = 1e-5  # train stops when lr less then this parameter


    # Model parameters
    input_channels = k  # k VMD components per signal sample
    num_classes = len(tx_list)  # Number of transmitters

    # Initialize model, loss, and optimizer
    model = CNNTransformerModel(input_channels=input_channels, num_classes=num_classes, split_channels=None).to(device)
    # model = CNNTransformerModel(input_channels=input_channels, num_classes=num_classes, split_channels=(5, 5)).to(device)


    # Initialize model, loss, and optimizer

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if enable_lr_reduction:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1) # default setting



    # --- Training and Evaluation ---

    # Training loop

    min_val_loss = 1e8
    current_lr = optimizer.param_groups[0]['lr']
    train_data = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }


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

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            with open(f"model_{name}",'wb') as f:
                pickle.dump(model, f)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Current LR: {current_lr}')
        train_data['train_loss'].append(train_loss)
        train_data['val_loss'].append(val_loss)
        train_data['lr'].append(current_lr)


        if enable_lr_reduction:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            print("Early stopping due to too low lr.")
            with open(f"train_dat_{name}",'wb') as f:
                pickle.dump(train_data, f)
            break


    # Testing
    with open(f"model_{name}",'rb') as f:
        model = pickle.load(f)
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
    print(classification_report(y_true, y_pred, target_names=tx_list,digits=5))
    return precision_score(y_true, y_pred, average='weighted'), \
    recall_score(y_true, y_pred, average='weighted'),\
    f1_score(y_true, y_pred, average='weighted'),\
    accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    precision_scores = [0,0,0,0,0]
    recall_scores = [0,0,0,0,0]
    f1_scores = [0,0,0,0,0]
    accuracy_scores = [0,0,0,0,0]
    for kfold_id in range(5):
        precision_scores[kfold_id], recall_scores[kfold_id], f1_scores[kfold_id], accuracy_scores[kfold_id] = train_eval(kfold_id)

    print("------- Overall report ---------")
    print(f"precision: {np.mean(precision_scores):.6f} (std err.:{np.std(precision_scores, ddof=1):.6f})")
    print(f"recall: {np.mean(recall_scores):.6f} (std err.:{np.std(recall_scores, ddof=1):.6f})")
    print(f"f1 score: {np.mean(f1_scores):.6f} (std err.:{np.std(f1_scores, ddof=1):.6f})") 
    print(f"accuracy: {np.mean(accuracy_scores):.6f} (std err.:{np.std(accuracy_scores, ddof=1):.6f})") 