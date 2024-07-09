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
import os
import wandb
import itertools
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv
import shutil
from utilities.Hyperparameters import arguments
from utilities.Networks import networks
from utilities.Data import get_filename
from utilities.utils import (
    train,
    dataframe,
    database,
    plot,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    predict_image_mask_miou,
    miou_score,
    pixel_acc,
    test_accuracy,
    check_accuracy,
    save_predictions_as_imgs,
    save_table,
    num_parameters
)

def create_csv(image_path, mask_path, X_train, X_val, X_test, filename='dataset.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Split', 'Image', 'Mask'])

        # Write training data
        for img_id in X_train:
            writer.writerow(['Train', f'{image_path}{img_id}.jpg', f'{mask_path}{img_id}.png'])

        # Write validation data
        for img_id in X_val:
            writer.writerow(['Validation', f'{image_path}{img_id}.jpg', f'{mask_path}{img_id}.png'])

        # Write test data
        for img_id in X_test:
            writer.writerow(['Test', f'{image_path}{img_id}.jpg', f'{mask_path}{img_id}.png'])

    print(f'Dataset CSV file created: {filename}')
    # Save the CSV file to wandb
    wandb.save(filename)


def clear_wandb_cache():
    wandb_dir = os.path.join(os.path.expanduser("~"), ".wandb")
    cache_dir = os.path.join(wandb_dir, "cache")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"Cleared wandb cache directory: {cache_dir}")
        except Exception as e:
            print(f"Error clearing wandb cache: {e}")
    else:
        print(f"Wandb cache directory not found: {cache_dir}")


def train_and_validate_model(args):
    # Your existing training and validation code here

    # Example placeholder code for validation loss
    val_loss = 0.0  # Replace with actual validation loss computation
    return val_loss


def main():
    # Define hyperparameter search space
    learning_rates = [1e-5, 1e-4, 1e-3]
    batch_sizes = [8, 16, 32]
    num_layers = [8, 12, 16]
    hidden_dims = [512, 768, 1024]
    num_heads = [8, 12, 16]
    dropout_rates = [0.1, 0.3, 0.5]
    patch_sizes = [8, 16, 32]

    best_val_loss = float('inf')
    best_params = None

    for lr, bs, nl, hd, nh, dr, ps in itertools.product(learning_rates, batch_sizes, num_layers, hidden_dims, num_heads,
                                                        dropout_rates, patch_sizes):
        args = arguments()
        args.lr = lr
        args.batch_size = bs
        args.num_layers = nl
        args.hidden_dim = hd
        args.num_heads = nh
        args.dropout_rate = dr
        args.patch_size = ps

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

        # Initialize wandb
        clear_wandb_cache()
        wandb.init(project="SemSeg", config=args)

        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on the: {device}")

        # datasets
        image_path, mask_path = database(args.data)

        df = dataframe(image_path)
        print(f'Total Images: {len(df)}')

        # Data split
        X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=args.random_state)
        X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=args.random_state)

        def check_data_split_overlap(X_train, X_val, X_test):
            train_set = set(X_train)
            val_set = set(X_val)
            test_set = set(X_test)

            assert train_set.isdisjoint(val_set), "Training and validation sets overlap!"
            assert train_set.isdisjoint(test_set), "Training and test sets overlap!"
            assert val_set.isdisjoint(test_set), "Validation and test sets overlap!"

            print("No overlap found between training, validation, and test sets.")

        check_data_split_overlap(X_train, X_val, X_test)

        print(f"Train size: {len(X_train)}")
        print(f"Validation size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")
        print(f"Random State: {args.random_state}")

        # Check unique labels in masks
        for split_name, split_data in zip(["Train", "Validation", "Test"], [X_train, X_val, X_test]):
            masks = [np.array(Image.open(mask_path + id + '.png')) for id in split_data]
            unique_labels = np.unique(masks)
            print(f"Unique labels in {split_name} masks:", unique_labels)

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

        create_csv(image_path, mask_path, X_train, X_val, X_test)

        model = networks(architecture=args.architecture, in_channels=args.in_channels, num_class=args.num_class,
                         config=config if args.architecture == 'unetr_2d' else None,
                         patch_size=args.patch_size if args.architecture in ['vitresunet', 'dilgabmpvitresunet'] else None).to(device)

        print(model)
        n_parameters = num_parameters(model)
        print(f"The model has {n_parameters:,} trainable parameters")
        criterion = nn.BCEWithLogitsLoss() if args.num_class == 1 else nn.CrossEntropyLoss()
        print(f"criterion: {criterion}")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_loader, val_loader, test_loader, test_ds = get_loaders(
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

        # Load saved model
        if args.load_model == 'True':
            print(f"Load model is {args.load_model}")
            if device == torch.device("cpu"):
                load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), model, optimizer)
            else:
                load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

        check_accuracy(val_loader, model, device=device, num_class=args.num_class)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            since = time.time()
            train(train_loader, model, optimizer, criterion, scaler, args.num_class, device)
            if args.save_model == 'True' and epoch % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)
            epoch_time_elapsed = time.time() - since
            print(f'Epoch {epoch} complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

        save_predictions_as_imgs(
            test_loader, model, folder="saved_images/", device=device
        )

        test_accuracy(test_loader, model, device, args.num_class)

        val_loss = train_and_validate_model(args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (lr, bs, nl, hd, nh, dr, ps)

    print("Best Parameters:", best_params)
    print("Best Validation Loss:", best_val_loss)

if __name__ == "__main__":
    main()
