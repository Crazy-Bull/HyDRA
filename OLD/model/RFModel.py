import torch
import torch.nn as nn
import math
from PER import PER
from MPDFormerEncoder import MPDFormerEncoder
from MultiPeriodicityFeatureFusion import MultiPeriodicityFeatureFusion
from MLPClassifier import MLPClassifier

class RFModel(nn.Module):
    def __init__(self, input_length, channels, periods, output_dims, num_heads, k, ff_dim, N, hidden_dim, fused_dim, num_classes, device):
        super(RFModel, self).__init__()
        self.per = PER(input_dim=(input_length, channels), periods=periods, output_dims=output_dims)
        self.encoders = nn.ModuleList([
            MPDFormerEncoder(embed_dim=embed_dim, num_heads=num_heads, p_i=p, k=k, ff_dim=ff_dim, N=N, N_i=N_i)
            for p, (N_i, embed_dim) in zip(periods, output_dims)
        ])
        self.fusion = MultiPeriodicityFeatureFusion(hidden_dim=hidden_dim, output_dim=fused_dim, device=device)
        mlp_input_dim = fused_dim * len(periods)
        self.mlp = MLPClassifier(input_dim=mlp_input_dim, output_dim=num_classes)

    def forward(self, x):
        embeddings = self.per(x)                          # List of embeddings for each period
        encoder_outputs = [encoder(emb) for encoder, emb in zip(self.encoders, embeddings)]
        fused = self.fusion(encoder_outputs)              # Fuse multi-periodicity features
        logits = self.mlp(fused)                          # Classify into num_classes
        return logits