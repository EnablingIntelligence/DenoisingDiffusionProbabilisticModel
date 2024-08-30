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

        x = torch.randn(16, 3, 256, 256)
        time_steps = torch.randint(0, 10, (16,))
        y = unet(x, time_steps)

        self.assertEquals(y.shape, (16, 3, 256, 256))


if __name__ == '__main__':
    unittest.main()
