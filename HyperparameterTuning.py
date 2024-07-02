import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import wandb
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv
import shutil
import sklearn
import pandas as pd
import optuna

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


def objective(trial):
    args = arguments()

    config = {
        "image_size": trial.suggest_categorical("image_size", [256, 512, 1024]),
        "num_layers": trial.suggest_int("num_layers", 1, 5),
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 512),
        "mlp_dim": trial.suggest_int("mlp_dim", 128, 1024),
        "num_heads": trial.suggest_int("num_heads", 2, 8),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "patch_size": trial.suggest_categorical("patch_size", [16, 32]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "num_patches": (args.image_size // args.patch_size) ** 2,
        "num_channels": args.in_channels
    }

    clear_wandb_cache()
    wandb.init(project="SemSeg", config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the: {device}")

    image_path, mask_path = database(args.data)
    df = dataframe(image_path)
    print(f'Total Images: {len(df)}')

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

    train_transform = A.Compose(
        [
            A.Resize(height=config["image_size"], width=config["image_size"]),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=config["image_size"], width=config["image_size"]),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(height=config["image_size"], width=config["image_size"]),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    create_csv(image_path, mask_path, X_train, X_val, X_test)

    model = networks(architecture=args.architecture, in_channels=args.in_channels, num_class=args.num_class,
                     config=config if args.architecture == 'unetr_2d' else None,
                     patch_size=config["patch_size"] if args.architecture in ['vitresunet', 'dilgabmpvitresunet'] else None).to(device)

    print(model)
    n_parameters = num_parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")

    if args.num_class == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    print(f"criterion: {criterion}")

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    train_loader, val_loader, test_loader, test_ds = get_loaders(
        image_path,
        mask_path,
        X_train,
        X_val,
        X_test,
        config["batch_size"],
        train_transform,
        val_transform,
        test_transform,
        args.num_workers,
        args.pin_memory,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        since = time.time()
        train(train_loader, model, optimizer, criterion, scaler, args.num_class, device)

        val_acc = check_accuracy(val_loader, model, device=device, num_class=args.num_class)
        if val_acc is None:
            val_acc = 0.0  # Ensure val_acc is a float

        print("Epoch:{}/{}..".format(epoch + 1, args.epochs),
              "Time: {:.2f}m".format((time.time() - since) / 60))

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=6000)  # Adjust n_trials and timeout as needed

    print("Best trial:")
    trial = study.best_trial

    print(f" Value: {trial.value}")
    print(f" Params: {trial.params}")
