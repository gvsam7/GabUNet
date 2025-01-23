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
from models.ResUNet18 import ResUNet18
from models.GabMPResUNet import GabMPResUNet
from models.DilResUNet import DilResUNet
from models.DilGabMPResUNet import DilGabMPResUNet
from models.DualDomAttnDilLogGabMPResUNet import DualDomAttnDilLogGabMPResUNet
from models.ViTResUNet import ViTResUNet
from models.ViTResUNet18 import ViTResUNet18
from models.DilGabMPResUNet18 import DilGabMPResUNet18
from models.DilGabMPViTResUNet import DilGabMPViTResUNet
from models.UNETR_2D import UNETR_2D
from models.SwinUNet import SwinUNet


def networks(architecture, in_channels, num_class, config=None, config2=None, patch_size=None):
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
    elif architecture == 'resunet18':
        model = ResUNet18(in_channels, num_class)
    elif architecture =='gabmpresunet':
        model = GabMPResUNet(in_channels, num_class)
    elif architecture == 'dilresunet':
        model = DilResUNet(in_channels, num_class)
    elif architecture == 'dilgabmpresunet':
        model = DilGabMPResUNet(in_channels, num_class)
    elif architecture == 'dualdomattndilloggabmpresunet':
        model = DualDomAttnDilLogGabMPResUNet(in_channels, num_class)
    elif architecture == 'vitresunet':
        model = ViTResUNet(in_channels, num_class, vit_patch_size=patch_size)
    elif architecture == 'vitresunet18':
        model = ViTResUNet18(in_channels, num_class)
    elif architecture == 'dilgabmpvitresunet':
        model = DilGabMPViTResUNet(in_channels, num_class, vit_patch_size=patch_size)
    elif architecture == 'swinunet':
        if config2 is None:
            raise ValueError("Config2 dictionary is required for SwinUNet model")
        model = SwinUNet(in_channels, num_class, config2)
    elif architecture == 'unetr_2d':
        if config is None:
            raise ValueError("Config dictionary is required for UNETR_2D model")
        model = UNETR_2D(in_channels, num_class, config)
    else:
        model = G_UNet(in_channels, num_class)
    return model