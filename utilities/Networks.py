import torchvision
from torch import nn
from models.UNet import UNet
from models.G_UNet import G_UNet


def networks(architecture, in_channels, out_channels):
    if architecture == 'unet':
        model = UNet(in_channels, out_channels)
    else:
        model = G_UNet(in_channels, out_channels)
    return model