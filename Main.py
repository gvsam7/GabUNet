import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from utilities.Hyperparameters import arguments
from models.UNet import UNet
from utilities.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_table
)

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    args = arguments()
    wandb.init(entity="predictive-analytics-lab", project="SemSeg")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the: {device}")

    train_transform = A.Compose(
        [
            A.Resize(height=args.height, width=args.width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=args.height, width=args.width),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        args.batch_size,
        train_transform,
        val_transforms,
        args.num_workers,
        args.pin_memory,
    )

    if args.load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, loss_fn, scaler, device)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=device)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=device
        )
        wandb.save(os.path.join('saved_images', '*'))

        save_table(val_loader, model, "Predictions", device)


if __name__ == "__main__":
    main()