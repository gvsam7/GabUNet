import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from models.ConvBlock import GConvBlock, ConvBlock, DilConvBlock


class G_UNet(nn.Module):
    def __init__(self, in_channels, num_class, features=[32, 64, 128, 256, 512]):
        super(G_UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # Down part of UNET
        for feature in features:
            if feature == features[0]:
                self.downs.append(GConvBlock(in_channels, feature))
            # Added dilated last conv layer
            elif feature == features[-1]:
                self.downs.append(DilConvBlock(in_channels, feature))
            else:
                self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(ConvBlock(feature * 2, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], num_class, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# def test():
#     x = torch.randn((3, 1, 161, 161))
#     model = UNet(in_channels=1, out_channels=1)
#     preds = model(x)
#     assert preds.shape == x.shape
#
#
# if __name__ == "__main__":
#     test()
