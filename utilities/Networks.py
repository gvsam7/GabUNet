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
from models.ViTResUNet import ViTResUNet
from models.ViTResUNet18 import ViTResUNet18
from models.UNETR_2D import UNETR_2D
from utilities.Hyperparameters import arguments



args = arguments()
config = {
        "image_size": args.image_size,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "mlp_dim": args.mlp_dim,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "num_patches": (args.image_size // args.patch_size) ** 2,
        "patch_size": args.patch_size,
        "num_channels": args.in_channels
    }


def networks(architecture, in_channels, num_class, config=None):
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
    elif architecture == 'vitresunet':
        model = ViTResUNet(in_channels, num_class)
    elif architecture == 'vitresunet18':
        model = ViTResUNet18(in_channels, num_class)
    elif architecture == 'unetr_2d':
        if config is None:
            raise ValueError("Config dictionary is required for UNETR_2D model")
        model = UNETR_2D(in_channels, num_class, config)
    else:
        model = G_UNet(in_channels, num_class)
    return model