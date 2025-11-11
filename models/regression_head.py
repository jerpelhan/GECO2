from torch import nn


class UpsamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(UpsamplingLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.reset_parameters()

    def forward(self, x):
        return self.layer(x)


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
