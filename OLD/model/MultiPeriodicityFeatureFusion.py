import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPeriodicityFeatureFusion(nn.Module):
    """A PyTorch module encapsulating feature fusion inspired by the MPDFormer encoder."""
    def __init__(self, hidden_dim=64, output_dim=128, device='cuda'):
        """
        Initialize the MultiPeriodicityFeatureFusion module.
        
        Args:
            hidden_dim (int): Hidden dimension for the MLP (default: 64).
            output_dim (int): Output dimension per input tensor after MLP (default: 128).
            device (str): Device to run the module on (default: 'cuda').
        """
        super(MultiPeriodicityFeatureFusion, self).__init__()
        self.device = device
        # MLP will be initialized dynamically based on input size in forward pass
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, E_list):
        """
            Process a list of input tensors through transposition, pooling, padding, weighting, and concatenation.
            
            Args:
                E_list (list of torch.Tensor): List of input tensors, each of shape [batch_size, N_i, embed_dim_i],
                                            where N_i is the number of periods and embed_dim_i is the embedding dimension.
            
            Returns:
                torch.Tensor: Concatenated output tensor of shape [batch_size, output_dim * len(E_list)].
            """
        # Move all tensors to the specified device
        E_list = [E.to(self.device) for E in E_list]

        # Step 1: Transpose each tensor to [batch_size, embed_dim_i, N_i]
        T_list = [E.transpose(1, 2) for E in E_list]

        # Step 2: Apply Global Average Pooling over the period dimension (dim=2)
        G_list = [T.mean(dim=2) for T in T_list]  # each [batch_size, embed_dim_i]

        # Step 3: Determine the maximum embedding dimension (N_max) and pad
        N_max = max(G.shape[1] for G in G_list)  # max embed_dim_i across all tensors
        G_padded = [F.pad(G, (0, N_max - G.shape[1])) for G in G_list]  # each [batch_size, N_max]

        # Step 4: Initialize and apply Adaptive Weighting (MLP) dynamically
        mlp = AdaptiveWeighting(input_dim=N_max, hidden_dim=self.hidden_dim, output_dim=self.output_dim).to(self.device)
        W_list = [mlp(G) for G in G_padded]  # each [batch_size, output_dim]

        # Step 5: Concatenate the MLP outputs along the feature dimension
        O = torch.cat(W_list, dim=1)  # [batch_size, output_dim * len(E_list)]
        return O

class AdaptiveWeighting(nn.Module):
    """A simple MLP for adaptive weighting of padded feature vectors."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the AdaptiveWeighting MLP.
        
        Args:
            input_dim (int): Input dimension (N_max from padding).
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output dimension.
        """
        super(AdaptiveWeighting, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Example Usage
if __name__ == "__main__":
    # Sample input tensors
    batch_size = 32
    E1 = torch.randn(batch_size, 29, 144)  # [batch_size, N_1, embed_dim_1]
    E2 = torch.randn(batch_size, 37, 122)  # [batch_size, N_2, embed_dim_2]
    E_list = [E1, E2]

    # Initialize the fusion module
    fusion_module = MultiPeriodicityFeatureFusion(hidden_dim=64, output_dim=128, device='cpu')

    # Forward pass
    O = fusion_module(E_list)
    print(f"Output shape: {O.shape}")  # Expected: torch.Size([32, 256])