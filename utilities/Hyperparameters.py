import argparse


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", default=True)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--architecture", default='unet', help='unet=UNet, gunet=G_UNet')

    return parser.parse_args()