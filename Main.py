"""
Author: Georgios Voulgaris
Date: 10/08/2022
Description: This project is aiming to research water detection using semantic segmentation.
             Data: The dataset is comprised of water body images (lakes, rivers) taken from unmanned aerial vehicles
             (UAV).
             The images were annotated using VGG Image Annotator (VIA). Then, from the created .json file masks, of the
             water bodies were created.
             Architecture: Initially, UNet is used to measure its performance, when trained on the water bodies dataset.
             Measurements are in the form of:
                1. visual inspection of the produced masks on test data,
                2. per pixel accuracy,
                3. Dice Score.
             Aim: This is going to be a test platform of testing various architectures.
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from utilities.Hyperparameters import arguments
from utilities.Networks import networks
from utilities.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_table,
    num_parameters
)

TRAIN_IMG_DIR = "Data/train_images/"
TRAIN_MASK_DIR = "Data/train_masks/"
VAL_IMG_DIR = "Data/val_images/"
VAL_MASK_DIR = "Data/val_masks/"


def train(loader, model, optimizer, criterion, scaler, num_class, device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        if num_class == 1:
            targets = targets.float().unsqueeze(1).to(device=device)
        else:
            targets = targets.long().to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    args = arguments()
    wandb.init(entity="predictive-analytics-lab", project="SemSeg", config=args)

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

    model = networks(architecture=args.architecture, in_channels=args.in_channels, num_class=args.num_class).to(device)
    print(model)
    n_parameters = num_parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")
    if args.num_class == 1:
        criterion = nn.BCEWithLogitsLoss()  # 1-class semantic segmentation
    else:
        criterion = nn.CrossEntropyLoss()
    print(f"criterion: {criterion}")
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

    # Load model
    if args.load_model == 'True':
        print(f"Load model is {args.load_model}")
        if device == torch.device("cpu"):
            load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), model, optimizer)
        else:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    check_accuracy(val_loader, model, device=device, num_class=args.num_class)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, criterion, scaler, args.num_class, device)

        # Saving model
        if args.save_model == 'True':
            if epoch % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=device, num_class=args.num_class)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=device)
        wandb.save(os.path.join('saved_images', '*'))

        save_table(val_loader, model, "Predictions", device)


if __name__ == "__main__":
    main()
