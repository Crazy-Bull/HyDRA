import torch.nn as nn
import torch

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim1=128, hidden_dim2=64, output_dim=32, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)         # [batch_size, 256] -> [batch_size, 128]
        x = self.selu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)         # [batch_size, 128] -> [batch_size, 64]
        x = self.selu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)         # [batch_size, 64] -> [batch_size, 32]
        return x

# Example usage
if __name__ == "__main__":
    mlp = MLPClassifier(input_dim=256, output_dim=32)
    input_tensor = torch.randn(16, 256)  # Batch of 16 samples
    logits = mlp(input_tensor)           # Output: [16, 32]
    print(f"Output shape: {logits.shape}")