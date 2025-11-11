import torch
import math
from PER import PER
from MPDFormerEncoder import MPDFormerEncoder
from MultiPeriodicityFeatureFusion import MultiPeriodicityFeatureFusion
from MLPClassifier import MLPClassifier

# Set device for computation (use CUDA if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define hyperparameters
batch_size = 32
input_length = 2048
channels = 2
f = [29, 37]
periods = []
for value in f:
    period = math.ceil(2048 / value)  # Calculate the ceiling of 2048 / f
    if period % 2 != 0:  # If the result is odd
        period += 1  # Round up to the nearest even number
    periods.append(period)

# Compute N_i and embed_dim_i for each period based on PER requirements
N_list = [math.ceil(input_length / p) for p in periods]  # N_i = ceil(input_length / period)
embed_dim_list = [2 * p for p in periods]  # embed_dim_i = 2 * period
output_dims = list(zip(N_list, embed_dim_list))  # e.g., [(29, 144), (37, 112)]

# Initialize the PER module
per = PER(input_dim=(input_length, channels), periods=periods, output_dims=output_dims).to(device)
print("PER module initialized")

# Define MPDFormerEncoder hyperparameters
num_heads = 8
k = 3
ff_dim = 512
N = 6  # Number of encoding layers

# Initialize a list of encoders, one for each periodicity
encoders = []
for p, N_i, embed_dim in zip(periods, N_list, embed_dim_list):
    encoder = MPDFormerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        p_i=p,
        k=k,
        ff_dim=ff_dim,
        N=N,
        N_i=N_i
    ).to(device)
    encoders.append(encoder)
print(f"Initialized {len(encoders)} MPDFormerEncoders")

# Initialize the Multi-Periodicity Feature Fusion module
fusion_module = MultiPeriodicityFeatureFusion(hidden_dim=64, output_dim=128, device=device)
print("Multi-Periodicity Feature Fusion module initialized")

# Initialize the MLP Classifier
mlp_input_dim = 128 * len(periods)  # output_dim * number of periods = 128 * 2 = 256
mlp = MLPClassifier(input_dim=mlp_input_dim, output_dim=32).to(device)
print("MLP Classifier initialized")

# Create a sample input tensor for testing
input_tensor = torch.randn(batch_size, input_length, channels).to(device)
print(f"Input tensor shape: {input_tensor.shape}")

# Forward pass through PER module
embeddings = per(input_tensor)
print("PER module output:")
for i, emb in enumerate(embeddings):
    print(f"  Embedding {i+1} shape: {emb.shape}")  # Expected: e.g., [32, 29, 144], [32, 37, 112]

# Forward pass through MPDFormerEncoders
encoder_outputs = []
for i, (encoder, emb) in enumerate(zip(encoders, embeddings)):
    output = encoder(emb)
    encoder_outputs.append(output)
    print(f"Encoder {i+1} output shape: {output.shape}")  # Expected: same as input, e.g., [32, 29, 144]

# Forward pass through Multi-Periodicity Feature Fusion module
fused = fusion_module(encoder_outputs)
print(f"Fused output shape: {fused.shape}")  # Expected: [32, 256]

# Forward pass through MLP Classifier
logits = mlp(fused)
print(f"Logits shape: {logits.shape}")  # Expected: [32, 32]