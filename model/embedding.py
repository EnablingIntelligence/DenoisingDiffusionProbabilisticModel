import torch
from torch import nn


class TimeEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.half_embedding_dim = embedding_dim // 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exponents = (
                torch.arange(0, self.half_embedding_dim, dtype=torch.float32) / self.halv_embedding_dim
        ).to(x.device)
        factors = 10_000 ** exponents

        embedding_arguments = x[:, None].repeat(1, self.half_embedding_dim) / factors
        embeddings = torch.cat([torch.sin(embedding_arguments), torch.cos(embedding_arguments)], dim=-1)
        return embeddings
