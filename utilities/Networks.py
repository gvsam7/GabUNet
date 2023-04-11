import torchvision
from torch import nn
from models.UNet import UNet
from models.G_UNet import G_UNet
from models.MAC_UNet import MAC_UNet
from models.GMAC_UNet import GMAC_UNet
from models.MixPGMAC_UNet import MixPGMAC_UNet
from models.DilMixPGMAC_UNet import DilMixPGMAC_UNet
from models.AsymUNet import AsymUNet
from models.ResUNet import ResUNet
from models.GabMPResUNet import GabMPResUNet
from models.DilResUNet import DilResUNet
from models.DilGabMPResUNet import DilGabMPResUNet


def networks(architecture, in_channels, num_class):
    if architecture == 'unet':
        model = UNet(in_channels, num_class)
    elif architecture == 'mac_unet':
        model = MAC_UNet(in_channels, num_class)
    elif architecture == 'gmac_unet':
        model = GMAC_UNet(in_channels, num_class)
    elif architecture == 'mixpgmac_unet':
        model = MixPGMAC_UNet(in_channels, num_class)
    elif architecture == 'dilmixpgmac_unet':
        model = DilMixPGMAC_UNet(in_channels, num_class)
    elif architecture == 'asymunet':
        model = AsymUNet(in_channels, num_class)
    elif architecture == 'resunet':
        model = ResUNet(in_channels, num_class)
    elif architecture =='gabmpresunet':
        model = GabMPResUNet(in_channels, num_class)
    elif architecture == 'dilresunet':
        model = DilResUNet(in_channels, num_class)
    elif architecture == 'dilgabmpresunet':
        model = DilGabMPResUNet(in_channels, num_class)
    else:
        model = G_UNet(in_channels, num_class)
    return model