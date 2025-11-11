import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import math
from mydataset import myDataset
from RFModel import RFModel
from tqdm import tqdm  # Import tqdm for progress bar

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dataset
path = r"H:\RF fingerprint\SingleDay.pkl/SingleDay.pkl"
dataset = myDataset(path)
print(f"Dataset size: {len(dataset)}")
print(f"Sample shape: {dataset[0][0].shape}, Label: {dataset[0][1]}")
num_classes = 28  # Number of devices
print(f"Number of classes: {num_classes}")

# Split dataset into train and test
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define hyperparameters
input_length = 256  # Matches dataset sample shape
channels = 2
f = [1.27, 2.54, 3.71, 4.98, 6.25, 7.53]
periods = []
for value in f:
    period = math.ceil(input_length / value)
    if period % 2 != 0:
        period += 1
    periods.append(period)
print(f"Periods: {periods}")

N_list = [math.ceil(input_length / p) for p in periods]
embed_dim_list = [2 * p for p in periods]
output_dims = list(zip(N_list, embed_dim_list))
print(f"Output dims: {output_dims}")

# Model hyperparameters
num_heads = 4
k = 3
ff_dim = 512
N = 2
hidden_dim = 64
fused_dim = 256  # Adjust based on MultiPeriodicityFeatureFusion output

# Initialize model
model = RFModel(
    input_length=input_length,
    channels=channels,
    periods=periods,
    output_dims=output_dims,
    num_heads=num_heads,
    k=k,
    ff_dim=ff_dim,
    N=N,
    hidden_dim=hidden_dim,
    fused_dim=fused_dim,
    num_classes=num_classes,
    device=device,
).to(device)
print("Model initialized")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = inputs.to(device)  # Shape: [batch_size, 256, 2]
        labels = labels.to(device)  # Shape: [batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)     # Shape: [batch_size, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")