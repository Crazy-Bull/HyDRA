import torch
import torch.nn as nn
import math

class PER(nn.Module):
    def __init__(self, input_dim, periods, output_dims):
        """
        Initialize the Periodicity Embedding Representation (PER) module.

        Args:
            input_dim (tuple): Input signal shape (length, channels), e.g., (2048, 2) for L=2048, I/Q channels.
            periods (list): List of periods derived from spectrum offset analysis, e.g., [72, 56].
            output_dims (list): Target output shapes [(N_1, d_model_1), ..., (N_k, d_model_k)], e.g., [(29, 144), (37, 112)].
        """
        super(PER, self).__init__()
        self.input_dim = input_dim  # (signal_len, channels)
        self.periods = periods      # e.g., [72, 56]
        self.output_dims = output_dims  # e.g., [(29, 144), (37, 112)]

        # Learnable projections to map 2 * period to target d_model for each period
        self.projections = nn.ModuleList([
            nn.Linear(2 * period, d_model) 
            for period, (_, d_model) in zip(periods, output_dims)
        ])

    def forward(self, x):
        """
        Transform the input signal into periodicity embeddings.

        Args:
            x (torch.Tensor): Input signal of shape (batch_size, 2048, 2) with I and Q channels.

        Returns:
            list[torch.Tensor]: List of embeddings, e.g., [(batch_size, 29, 144), (batch_size, 37, 112)].
        """
        batch_size, signal_len, channels = x.size()  # e.g., (batch_size, 2048, 2)
        assert signal_len == self.input_dim[0] and channels == self.input_dim[1], "Input shape mismatch"
        embeddings = []

        for i, period in enumerate(self.periods):
            # Step 1: Compute number of segments (N_i = ceil(L / p_i))
            n_segments = math.ceil(signal_len / period)  # e.g., ceil(2048 / 72) = 29
            assert n_segments == self.output_dims[i][0], f"Expected {self.output_dims[i][0]} segments, got {n_segments}"
            d_model = self.output_dims[i][1]  # Target embedding dim, e.g., 144

            # Step 2: Pad signal if necessary to fit n_segments * period
            total_len = n_segments * period  # e.g., 29 * 72 = 2088
            pad_size = total_len - signal_len  # e.g., 2088 - 2048 = 40
            if pad_size > 0:
                padding = torch.zeros(batch_size, pad_size, channels, device=x.device)
                x_padded = torch.cat([x, padding], dim=1)  # (batch_size, 2088, 2)
            else:
                x_padded = x  # No padding needed

            # Step 3: Segment the signal into (batch_size, n_segments, period, channels)
            segments = x_padded.view(batch_size, n_segments, period, channels)  # e.g., (batch_size, 29, 72, 2)

            # Step 4: Concatenate I and Q channels into (batch_size, n_segments, 2 * period)
            embedding = segments.reshape(batch_size, n_segments, -1)  # e.g., (batch_size, 29, 144)

            # Step 5: Apply learnable projection to adjust to target d_model
            embedding = self.projections[i](embedding)  # e.g., (batch_size, 29, 144)

            embeddings.append(embedding)

        return embeddings

# Example usage:
if __name__ == "__main__":
    # Define parameters
    input_dim = (2048, 2)
    periods = [72, 56]
    output_dims = [(29, 144), (37, 112)]
    
    # Initialize PER module
    per = PER(input_dim, periods, output_dims)
    
    # Create sample input
    batch_size = 4
    x = torch.randn(batch_size, 2048, 2)
    
    # Forward pass
    embeddings = per(x)
    
    # Check output shapes
    for i, emb in enumerate(embeddings):
        print(f"Embedding {i + 1} shape: {emb.shape}")  # e.g., (4, 29, 144), (4, 37, 112)