import argparse


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", default=True)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-class", type=int, default=1)
    parser.add_argument("--data", default='landcover_ai', help='water, landcover_ai, WHDLD, uavid, treecrown, '
                                                               'treecrowncanada, treecrown_ndvi')
    parser.add_argument("--saved-images", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    # parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--save-model", default=False)
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--architecture", default='unet', help='unet=UNet, gunet=G_UNet, mac_unet=MAC_UNet, '
                                                               'gmac_unet=GMAC_UNet, mixpgmac_unet=MixPGMAC_UNet,'
                                                               'asymunet=AsymUNet, dilmixpgmac_unet=DilMixPGMAC_UNet,'
                                                               'resunet=ResUNet, resunet18=ResUNet18,'
                                                               'dilgabmpresunet18=DilGabMPResUNet18,'
                                                               'gabmpresunet=GabMPResUNet,'
                                                               'dilresunnet=DilResUNet, dilgabmpresunet=DilGabMPResUNet,'
                                                               'dualdomattndilloggabmpresunet=DualDomAttnDilLogGabMPResUNet'
                                                               'vitresunet=VitResUNet, vitresunet18=ViTResUNet18,'
                                                               'swinunet=SwinUNet, deeplabveplus=DeepLabV3Plus'
                                                               'unetr_2d=UNETR_2D, dilgabmpvitresunet=DilGabMPViTResUNet,'
                                                               'transunet=TransUNet')
    # Transformer
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--mlp-dim", type=int, default=3072)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=16)

    return parser.parse_args()