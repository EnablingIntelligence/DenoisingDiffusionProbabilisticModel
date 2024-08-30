import torch
from torch import nn

from utils import repeat_layers


class Encoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, num_layers: int, num_heads: int,
                 reduce_size: bool):
        super().__init__()
        self.num_layers = num_layers

        self.embedding_layers = repeat_layers(
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=embedding_dim, out_features=out_channels)
            ),
            num_layers
        )

        self.conv1_layers = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels if layer_idx == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels if layer_idx == 0 else out_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for layer_idx in range(num_layers)
        ])
        self.conv2_layers = repeat_layers(
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            ),
            num_layers
        )
        self.conv_residuals = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels if layer_idx == 0 else out_channels, out_channels=out_channels,
                      kernel_size=1)
            for layer_idx in range(num_layers)
        ])
        self.down_sample_layer = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                           padding=1) if reduce_size else nn.Identity()

        self.attention_norms = repeat_layers(nn.GroupNorm(num_groups=8, num_channels=out_channels), num_layers)
        self.attentions = repeat_layers(
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True),
            num_layers
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.num_layers):
            residual_input = x
            x = self.conv1_layers[layer_idx](x)
            x = x + self.embedding_layers[layer_idx](time_embedding)[:, :, None, None]
            x = self.conv2_layers[layer_idx](x)
            x = x + self.conv_residuals[layer_idx](residual_input)

            batch_size, channels, h, w = x.shape
            x_att = x.reshape(batch_size, channels, h * w)
            x_att = self.attention_norms[layer_idx](x_att).transpose(1, 2)
            x_att, _ = self.attentions[layer_idx](x_att, x_att, x_att).transpose(1, 2)
            x_att = x_att.reshape(batch_size, channels, h, w)
            x = x + x_att

        return self.down_sample_layer(x)


class Bottleneck(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.num_layers = num_layers

        self.embedding_layers = repeat_layers(
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=embedding_dim, out_features=out_channels)
            ),
            num_layers + 1
        )

        self.conv1_layers = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels if layer_idx == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels if layer_idx == 0 else out_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for layer_idx in range(num_layers + 1)
        ])
        self.conv2_layers = repeat_layers(
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            ),
            num_layers + 1
        )
        self.conv_residuals = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels if layer_idx == 0 else out_channels, out_channels=out_channels,
                      kernel_size=1)
            for layer_idx in range(num_layers + 1)
        ])

        self.attention_norms = repeat_layers(nn.GroupNorm(num_groups=8, num_channels=out_channels), num_layers)
        self.attention = repeat_layers(
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True),
            num_layers
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual_input = x

        x = self.conv1_layers[0](x)
        x = x + self.embedding_layers[0](time_embedding)[:, :, None, None]
        x = self.conv2_layers[0](x)
        x = x + self.conv_residuals[0](residual_input)

        for layer_idx in range(self.num_layers):
            batch_size, channels, h, w = x.shape
            x_att = x.reshape(batch_size, channels, h * w)
            x_att = self.attention_norms[layer_idx](x_att).transpose(1, 2)
            x_att, _ = self.attention[layer_idx](x_att, x_att, x_att).transpose(1, 2)
            x_att = x_att.reshape(batch_size, channels, h, w)
            x = x + x_att

            residual_input = x
            x = self.conv1_layers[layer_idx + 1](x)
            x = x + self.embedding_layers[layer_idx + 1](time_embedding)[:, :, None, None]
            x = self.conv2_layers[layer_idx + 1](x)
            x = x + self.conv_residuals[layer_idx + 1](residual_input)

        return x


class Decoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, num_layers: int, num_heads: int,
                 increase_size: bool):
        super().__init__()
        self.num_layers = num_layers

        self.embedding_layers = repeat_layers(
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=embedding_dim, out_features=out_channels)
            ),
            num_layers
        )

        self.conv1_layers = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels if layer_idx == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels if layer_idx == 0 else out_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for layer_idx in range(num_layers)
        ])
        self.conv2_layers = repeat_layers(
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            ),
            num_layers
        )
        self.conv_residuals = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels if layer_idx == 0 else out_channels, out_channels=out_channels,
                      kernel_size=1)
            for layer_idx in range(num_layers)
        ])
        self.up_sample_layer = nn.ConvTranspose2d(in_channels=in_channels // 2, out_channels=in_channels // 2,
                                                  kernel_size=4, stride=2,
                                                  padding=1) if increase_size else nn.Identity()

        self.attention_norms = repeat_layers(nn.GroupNorm(num_groups=8, num_channels=out_channels), num_layers)
        self.attention = repeat_layers(
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True),
            num_layers
        )

    def forward(self, x: torch.Tensor, out_encoder: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = self.up_sample_layer(x)
        x = torch.cat([x, out_encoder], dim=1)

        for layer_idx in range(self.num_layers):
            residual_input = x
            x = self.conv1_layers[layer_idx](x)
            x = x + self.embedding_layers[layer_idx](time_embedding)[:, :, None, None]
            x = self.conv2_layers[layer_idx](x)
            x = x + self.conv_residuals[layer_idx](residual_input)

            batch_size, channels, h, w = x.shape
            x_att = x.reshape(batch_size, channels, h * w)
            x_att = self.attention_norms[layer_idx](x_att).transpose(1, 2)
            x_att, _ = self.attentions[layer_idx](x_att, x_att, x_att).transpose(1, 2)
            x_att = x_att.reshape(batch_size, channels, h, w)
            x = x + x_att

        return x
