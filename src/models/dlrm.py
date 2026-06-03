"""DLRM recommendation model for MLPerf inference on ROCm."""

import torch
import torch.nn as nn


class DLRM(nn.Module):
    """Deep Learning Recommendation Model."""

    def __init__(self, num_features=13, embedding_dim=128, num_embeddings=100000,
                 bottom_mlp_dims=[512, 256, 128], top_mlp_dims=[512, 512, 256, 1]):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Bottom MLP for dense features
        layers = []
        prev_dim = num_features
        for dim in bottom_mlp_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.bottom_mlp = nn.Sequential(*layers)

        # Embedding tables for sparse features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) for _ in range(26)
        ])

        # Top MLP for combined features
        interaction_dim = bottom_mlp_dims[-1] + 26 * embedding_dim + bottom_mlp_dims[-1]
        top_layers = []
        prev_dim = interaction_dim
        for dim in top_mlp_dims[:-1]:
            top_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        top_layers.append(nn.Linear(prev_dim, top_mlp_dims[-1]))
        top_layers.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, dense, sparse):
        # Bottom MLP
        dense_out = self.bottom_mlp(dense)

        # Embedding lookup
        sparse_out = [emb(sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        sparse_out = torch.cat(sparse_out, dim=1)

        # Feature interaction
        combined = torch.cat([dense_out, sparse_out, dense_out], dim=1)

        # Top MLP
        return self.top_mlp(combined)


def load_dlrm(device: torch.device, precision: str = "fp32") -> nn.Module:
    """Load DLRM model."""
    model = DLRM().to(device)
    if precision == "fp16":
        model = model.half()
    model.eval()
    return model
