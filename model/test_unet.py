import unittest

import torch

from model import UNet, UNetConfig


class TestUNet(unittest.TestCase):

    def test_shapes(self):
        # Test the building and output shapes of the UNet model

        config = UNetConfig(
            in_channels=3,
            embedding_dim=128,
            encoder_channels=[32, 64, 128, 256],
            encoder_down_sample=[True, True, False],
            encoder_num_layers=2,
            bottleneck_channels=[256, 256, 128],
            bottleneck_num_layers=2,
            decoder_num_layers=2
        )
        unet = UNet(config)

        x = torch.randn(1, 3, 256, 256)
        embeddings = torch.randn(1, 128)
        y = unet(x, embeddings)

        self.assertEquals(y.shape, (1, 3, 256, 256))


if __name__ == '__main__':
    unittest.main()
