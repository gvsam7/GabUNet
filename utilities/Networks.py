import torchvision
from torch import nn
from models.UNet import UNet
from models.G_UNet import G_UNet
from models.MAC_UNet import MAC_UNet
from models.GMAC_UNet import GMAC_UNet
from models.MixPGMAC_UNet import MixPGMAC_UNet


def networks(architecture, in_channels, out_channels):
    if architecture == 'unet':
        model = UNet(in_channels, out_channels)
    elif architecture == 'mac_unet':
        model = MAC_UNet(in_channels, out_channels)
    elif architecture == 'gmac_unet':
        model = GMAC_UNet(in_channels, out_channels)
    elif architecture == 'mixpgmac_unet':
        model = MixPGMAC_UNet(in_channels, out_channels)
    else:
        model = G_UNet(in_channels, out_channels)
    return model