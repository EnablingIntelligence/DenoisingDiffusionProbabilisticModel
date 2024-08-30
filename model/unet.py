from dataclasses import dataclass

import torch
from torch import nn

from model import TimeEmbedding, Encoder, Decoder, Bottleneck


@dataclass
class UNetConfig:
    in_channels: int
    embedding_dim: int

    encoder_channels: list[int]
    encoder_down_sample: list[bool]
    encoder_num_layers: int

    bottleneck_channels: list[int]
    bottleneck_num_layers: int

    decoder_num_layers: int


class UNet(nn.Module):

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim

        self.embedding = TimeEmbedding(embedding_dim=config.embedding_dim)
        self.embedding_proj = nn.Sequential(
            nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim),
            nn.SiLU(),
            nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim)
        )

        self.conv_input = nn.Conv2d(in_channels=config.in_channels, out_channels=config.encoder_channels[0],
                                    kernel_size=3, padding=1)

        self.up_sample = list(reversed(self.encoder_down_sample))
        self.encoder = nn.ModuleList([])
        for block_idx in range(len(config.encoder_channels) - 1):
            self.encoder.append(Encoder(in_channels=config.encoder_channels[block_idx],
                                        out_channels=config.encoder_channels[block_idx + 1],
                                        embedding_dim=config.embedding_dim, num_layers=config.encoder_num_layers,
                                        reduce_size=config.encoder_down_sample[block_idx], num_heads=4))

        self.bottleneck = nn.ModuleList([])
        for block_idx in range(len(config.bottleneck_channels) - 1):
            self.bottleneck.append(Bottleneck(in_channels=config.bottleneck_channels[block_idx],
                                              out_channels=config.bottleneck_channels[block_idx + 1],
                                              embedding_dim=config.embedding_dim,
                                              num_layers=config.bottleneck_num_layers,
                                              num_heads=4))

        self.decoder = nn.ModuleList([])
        for block_idx in reversed(range(len(config.encoder_channels) - 1)):
            self.decoder.append(Decoder(in_channels=config.encoder_channels[block_idx] * 2,
                                        out_channels=config.encoder_channels[block_idx - 1] if block_idx != 0 else 16,
                                        embedding_dim=config.embedding_dim, num_layers=config.decoder_num_layers,
                                        increase_size=self.up_sample[block_idx], num_heads=4))

        self.out_proj = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=config.in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(x)

        embedding = self.embedding(time_embedding)
        embedding = self.embedding_proj(embedding)

        encoder_outs = []
        for layer_idx, encoder_layer in enumerate(self.encoder):
            encoder_outs.append(x)
            x = encoder_layer(x, embedding)

        for bottleneck_layer in self.bottleneck:
            x = bottleneck_layer(x, embedding)

        for decoder_layer in self.decoder:
            x = decoder_layer(x, encoder_outs.pop(), embedding)

        return self.out_proj(x)
