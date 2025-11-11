import torch
import torch.nn as nn
import math

# Positional Encoding Function
def positional_encoding(N_i, p_i, device):
    """
    Compute positional encodings to add sequential order information to the input.

    Args:
        N_i (int): Number of periods (rows in the input).
        p_i (int): Period length.
        device (torch.device): Device for computation (e.g., 'cuda').

    Returns:
        torch.Tensor: Positional encoding tensor of shape [N_i, 2 * p_i].
    """
    pos = torch.arange(N_i, dtype=torch.float, device=device).unsqueeze(1)  # [N_i, 1]
    i = torch.arange(p_i, dtype=torch.float, device=device)  # [p_i]
    angle_rates = 1 / (10000 ** (i / p_i))  # [p_i]
    angle_rads = pos * angle_rates  # [N_i, p_i]
    pe = torch.zeros(N_i, 2 * p_i, device=device)  # [N_i, 2 * p_i]
    pe[:, 0::2] = torch.sin(angle_rads)  # Even indices
    pe[:, 1::2] = torch.cos(angle_rads)  # Odd indices
    return pe

# Projection and Scaling Module
class ProjectionAndScaling(nn.Module):
    def __init__(self, embed_dim):
        """
        Linear projection and scaling layer for input transformation.

        Args:
            embed_dim (int): Embedding dimension (2 * p_i).
        """
        super(ProjectionAndScaling, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)  # Linear projection
        self.scale = math.sqrt(embed_dim)  # Scaling factor based on embedding dimension

    def forward(self, x):
        """
        Forward pass for projection and scaling.

        Args:
            x (torch.Tensor): Input tensor of shape [N_i, embed_dim].

        Returns:
            torch.Tensor: Projected and scaled tensor of shape [N_i, embed_dim].
        """
        projected = self.linear(x)  # Linear transformation
        return projected / self.scale  # Apply scaling

# Multi-Head Inter-Period Attention
class MultiHeadInterPeriodAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, p_i, dropout=0.1):
        """
        Multi-head attention mechanism for inter-period dependencies.

        Args:
            embed_dim (int): Embedding dimension (2 * p_i).
            num_heads (int): Number of attention heads.
            p_i (int): Period length.
        """
        super(MultiHeadInterPeriodAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.p_i = p_i

        # Linear projections for Q, K, V (already projected by ProjectionAndScaling)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)  # Output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for inter-period attention.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, N_i, embed_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, N_i, embed_dim].
        """
        batch_size, N_i, embed_dim = x.shape
        # Project and reshape into multi-head format
        Q = self.W_q(x).view(batch_size, N_i, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, N_i, head_dim]
        K = self.W_k(x).view(batch_size, N_i, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(x).view(batch_size, N_i, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, N_i, N_i]
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, N_i, head_dim]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, N_i, embed_dim)
        output = self.W_o(attention_output)
        output = self.dropout(output)
        return output

# Multi-Head Intra-Period Attention
class MultiHeadIntraPeriodAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, p_i, k, dropout=0.1):
        """
        Multi-head attention mechanism for intra-period dependencies using DFT.

        Args:
            embed_dim (int): Embedding dimension (2 * p_i).
            num_heads (int): Number of attention heads.
            p_i (int): Period length.
            k (int): Number of top delays to consider.
        """
        super(MultiHeadIntraPeriodAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.p_i = p_i
        self.k = k

        # Linear projections for Q, K, V (already projected by ProjectionAndScaling)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)  # Output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for intra-period attention.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, N_i, embed_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, N_i, embed_dim].
        """
        batch_size, N_i, embed_dim = x.shape
        Q = self.W_q(x).view(batch_size, N_i, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, N_i, head_dim]
        K = self.W_k(x).view(batch_size, N_i, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(x).view(batch_size, N_i, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute cross-correlation in frequency domain with scaling
        Q_fft = torch.fft.fft(Q, dim=2)  # [num_heads, N_i, head_dim]
        K_fft = torch.fft.fft(K, dim=2)
        S_QK_Intra = Q_fft * torch.conj(K_fft)  # Element-wise multiplication
        C_QK_Intra = torch.fft.ifft(S_QK_Intra, dim=2).real  # [batch_size, num_heads, N_i, head_dim]

        # Average over head_dim and heads to get a single correlation per τ, with scaling
        C_QK_Intra_mean = C_QK_Intra.mean(dim=-1) / math.sqrt(self.head_dim)  # [batch_size, num_heads, N_i]
        C_QK_Intra_tau = C_QK_Intra_mean.mean(dim=1)  # [batch_size, N_i]

        # Select top-k delays (τ), limited by min(N_i, p_i)
        max_tau = min(N_i, self.p_i)
        _, top_k_indices = torch.topk(C_QK_Intra_tau[:, :max_tau], min(self.k, max_tau), dim=1)  # [batch_size, k]

        # Compute attention output by rolling V and weighting
        attention_output = torch.zeros_like(V)
        for i in range(min(self.k, max_tau)):
            tau = top_k_indices[:, i]  # [batch_size]
            V_rolled = torch.stack([torch.roll(V[b], shifts=int(tau[b]), dims=1) for b in range(batch_size)], dim=0)
            C_QK_Intra_selected = torch.stack([C_QK_Intra_mean[b, :, tau[b]] for b in range(batch_size)], dim=0)  # [batch_size, num_heads]
            weights = torch.softmax(C_QK_Intra_selected, dim=1)  # [batch_size, num_heads]
            attention_output += V_rolled * weights.unsqueeze(-1).unsqueeze(-1)  # Broadcasting
        attention_output /= min(self.k, max_tau)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, N_i, -1)
        output = self.W_o(attention_output)
        output = self.dropout(output)
        return output

# Attention Fusion
class AttentionFusion(nn.Module):
    def __init__(self):
        """
        Fuses inter-period and intra-period attention outputs with a residual connection.
        """
        super(AttentionFusion, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1.0))  # Weight for inter-period attention
        self.zeta = nn.Parameter(torch.tensor(1.0))   # Weight for intra-period attention

    def forward(self, X, Lambda, Sigma):
        """
        Forward pass for attention fusion.
        Args:
            X (torch.Tensor): Input tensor [batch_size, N_i, embed_dim].
            Lambda (torch.Tensor): Inter-period attention output [batch_size, N_i, embed_dim].
            Sigma (torch.Tensor): Intra-period attention output [batch_size, N_i, embed_dim].
        Returns:
            torch.Tensor: Fused output [batch_size, N_i, embed_dim].
        """
        return X + self.sigma * Lambda + self.zeta * Sigma

# Feed-Forward Network (FFN)
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Position-wise feed-forward network.

        Args:
            embed_dim (int): Input and output dimension.
            ff_dim (int): Hidden layer dimension.
        """
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for FFN.
        Args:
            x (torch.Tensor): Input tensor [batch_size, N_i, embed_dim].
        Returns:
            torch.Tensor: Output tensor [batch_size, N_i, embed_dim].
        """
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.dropout(x) # Add dropout
        return x

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, p_i, k, ff_dim, dropout=0.1):
        """
        Single encoding layer of the MPDFormer.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            p_i (int): Period length.
            k (int): Number of top delays in intra-period attention.
            ff_dim (int): FFN hidden dimension.
        """
        super(EncoderLayer, self).__init__()
        self.inter_attention = MultiHeadInterPeriodAttention(embed_dim, num_heads, p_i)
        self.intra_attention = MultiHeadIntraPeriodAttention(embed_dim, num_heads, p_i, k)
        self.fusion = AttentionFusion()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the encoder layer.
        Args:
            x (torch.Tensor): Input tensor [batch_size, N_i, embed_dim].
        Returns:
            torch.Tensor: Output tensor [batch_size, N_i, embed_dim].
        """
        # Apply projection and scaling before attention
        inter_output = self.inter_attention(x)
        intra_output = self.intra_attention(x)
        fused = self.fusion(x, inter_output, intra_output)
        norm1 = self.norm1(fused)
        ffn_output = self.ffn(norm1)
        output = self.norm2(norm1 + ffn_output)
        return output

# MPDFormer Encoder
class MPDFormerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, p_i, k, ff_dim, N, N_i):
        """
        Full MPDFormer encoder with N stacked layers.

        Args:
            embed_dim (int): Embedding dimension (2 * p_i).
            num_heads (int): Number of attention heads.
            p_i (int): Period length.
            k (int): Number of top delays in intra-period attention.
            ff_dim (int): FFN hidden dimension.
            N (int): Number of encoder layers.
            N_i (int): Number of periods.
        """
        super(MPDFormerEncoder, self).__init__()
        self.projection = ProjectionAndScaling(embed_dim)  # Add projection here
        self.pos_encoding = positional_encoding(N_i, p_i, device='cuda')
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, p_i, k, ff_dim) for _ in range(N)
        ])

    def forward(self, x):
        """
        Forward pass for the MPDFormer encoder.
        Args:
            x (torch.Tensor): Input tensor [batch_size, N_i, embed_dim].
        Returns:
            torch.Tensor: Output tensor [batch_size, N_i, embed_dim].
        """
        x = self.projection(x)  # Apply projection and scaling first
        x = x + self.pos_encoding.to(x.device)  # Then add positional encoding
        for layer in self.layers:
            x = layer(x)
        return x

# Example Usage
if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 32
    embed_dim = 144  # 2 * p_i
    num_heads = 8
    p_i = 72
    k = 3
    ff_dim = 512
    N = 6
    N_i = 29
    
    embed_dim2 = 112  # 2 * p_i
    p_i2 = 56
    N_i2 = 37

    # Instantiate the encoder
    encoder = MPDFormerEncoder(embed_dim, num_heads, p_i, k, ff_dim, N, N_i)
    encoder = encoder.to('cuda')
    
    encoder2 = MPDFormerEncoder(embed_dim2, num_heads, p_i2, k, ff_dim, N, N_i2)
    encoder2 = encoder2.to('cuda')

    # Create a sample input tensor
    input_tensor = torch.randn(batch_size, N_i, embed_dim).to('cuda')
    print(f"Input shape: {input_tensor.shape}")
    input_tensor2 = torch.randn(batch_size, N_i2, embed_dim2).to('cuda')
    print(f"Input shape2: {input_tensor2.shape}")

    # Forward pass
    output = encoder(input_tensor)
    print(f"Output shape: {output.shape}")
    output2 = encoder2(input_tensor2)
    print(f"Output shape2: {output2.shape}")