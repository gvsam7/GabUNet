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
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
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

# TRAIN_IMG_DIR = "Data_test/train_images/"
# TRAIN_MASK_DIR = "Data_test/train_masks/"
# VAL_IMG_DIR = "Data_test/val_images/"
# VAL_MASK_DIR = "Data_test/val_masks/"
# IMAGE_PATH = "Data_test/train_images/"
# MASK_PATH = "Data_test/train_masks/"


def dataframe(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            # name.append(filename.split('.')[0]  # for <name>.jpg images
            name.append(filename.rsplit('.', 1)[0])  # for <name>.<name>.jpg images

        return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


def database(data):
    if data == "landcover_ai":
        image_path = "Data_test/train_images/"
        mask_path = "Data_test/train_masks/"
    else:
        image_path = "Data/train_images/"
        mask_path = "Data/train_masks/"
    print(f"Dataset: {data}")
    return image_path, mask_path


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

    # datasets
    image_path, mask_path = database(args.data)
    print(f"image path: {image_path}")
    print(f"mask_path: {mask_path}")

    df = dataframe(image_path)
    print(f'Total Images: {len(df)}')

    # Data split
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=21)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=21)

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # img = Image.open(IMAGE_PATH + df['id'][39] + '.JPG')
    # mask = Image.open(MASK_PATH + df['id'][39] + '.PNG')
    img = Image.open(image_path + df['id'][39] + '.JPG')
    mask = Image.open(mask_path + df['id'][39] + '.PNG')
    print(f"Image size: {np.asarray(img).shape}")
    print(f"Mask size: {np.asarray(mask).shape}")

    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    plt.title('Image with Applied Masks')
    plt.show()

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

    val_transform = A.Compose(
        [
            A.Resize(height=args.height, width=args.width),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    test_transform = A.Compose(
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

    train_loader, val_loader, test_loader = get_loaders(
        # TRAIN_IMG_DIR,
        # TRAIN_MASK_DIR,
        # VAL_IMG_DIR,
        # VAL_MASK_DIR,
        # IMAGE_PATH,
        # MASK_PATH,
        image_path,
        mask_path,
        X_train,
        X_val,
        X_test,
        args.batch_size,
        train_transform,
        val_transform,
        test_transform,
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

        train(val_loader, model, optimizer, criterion, scaler, args.num_class, device)

        # check accuracy
        check_accuracy(test_loader, model, device=device, num_class=args.num_class)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, num_class=args.num_class, folder="saved_images/", device=device)
        wandb.save(os.path.join('saved_images', '*'))

        save_table(val_loader, args.num_class, model,  "Predictions", device)


if __name__ == "__main__":
    main()
