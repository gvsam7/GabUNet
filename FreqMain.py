"""
Author: Georgios Voulgaris
Date: 16/07/2024
Description: This function is to test the Enhanced Frequency Log-Gabor convolutional neural network.
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
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv
import shutil
import sklearn
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utilities.Hyperparameters import arguments
from utilities.Networks import networks
from utilities.Data import get_filename
from utilities.utils import (
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

# Check environment version
# print(f"scikit-learn version: {sklearn.__version__}")
# print(f"NumPy version: {np.__version__}")
# print(f"Pandas version: {pd.__version__}")


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


# Add these functions outside the training loop (16_07_2024)
def l1_regularization(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm


def freq_train(loader, model, optimizer, criterion, scaler, num_class, device, scheduler):
    loop = tqdm(loader)
    """
def train(loader, model, optimizer, criterion, scaler, gradient_accumulations, num_class, device):
    loop = tqdm(loader)
    """

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        if num_class == 1:
            targets = targets.float().unsqueeze(1).to(device=device)
        else:
            targets = targets.long().to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # loss = criterion(predictions, targets)
            # Calculate loss with L1 regularisation (16_07_2024)
            l1_lambda = 1e-5  # Adjust as needed
            loss = criterion(predictions, targets) + l1_regularization(model, l1_lambda)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(loss)

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        """
        scaler.scale(loss / gradient_accumulations).backward()

        if (batch_idx + 1) % gradient_accumulations == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            """

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
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

    # Clear wandb cache
    clear_wandb_cache()
    # wandb.init(entity="predictive-analytics-lab", project="SemSeg", config=args)
    wandb.init(project="SemSeg", config=args)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the: {device}")

    # datasets
    image_path, mask_path = database(args.data)
    # print(f"image path: {image_path}")
    # print(f"mask_path: {mask_path}")

    df = dataframe(image_path)
    print(f'Total Images: {len(df)}')

    # Data split
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=args.random_state)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=args.random_state)

    # Check for overlap between training, validation, and test sets
    def check_data_split_overlap(X_train, X_val, X_test):
        train_set = set(X_train)
        val_set = set(X_val)
        test_set = set(X_test)

        assert train_set.isdisjoint(val_set), "Training and validation sets overlap!"
        assert train_set.isdisjoint(test_set), "Training and test sets overlap!"
        assert val_set.isdisjoint(test_set), "Validation and test sets overlap!"

        print("No overlap found between training, validation, and test sets.")

    # Call the function to check data splits
    check_data_split_overlap(X_train, X_val, X_test)

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    print(f"Random State: {args.random_state}")

    """
    img = Image.open(image_path + df['id'][39] + '.jpg')
    mask = Image.open(mask_path + df['id'][39] + '.png')
    print(f"Image size: {np.asarray(img).shape}")
    print(f"Mask size: {np.asarray(mask).shape}")

    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    plt.title('Image with Applied Masks')
    # plt.show()
    """

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

    # Create a CSV listing all train, validation, and test images
    create_csv(image_path, mask_path, X_train, X_val, X_test)

    # model = networks(architecture=args.architecture, in_channels=args.in_channels, num_class=args.num_class,
    #                  config=config if args.architecture == 'unetr_2d' else None).to(device)
    model = networks(architecture=args.architecture, in_channels=args.in_channels, num_class=args.num_class,
                     config=config if args.architecture == 'unetr_2d' else None,
                     patch_size=args.patch_size if args.architecture in ['vitresunet', 'dilgabmpvitresunet'] else None).to(device)

    print(model)
    n_parameters = num_parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")
    if args.num_class == 1:
        criterion = nn.BCEWithLogitsLoss()  # 1-class semantic segmentation
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"criterion: {criterion}")

    # Implement layer-wise learning rates: Modify your optimiser setup
    params = [
        {'params': model.conv11.parameters(), 'lr': 1e-4},
        {'params': model.res2.parameters()},
        {'params': model.res3.parameters()},
        {'params': model.res4.parameters()},
        {'params': model.dec1.parameters()},
        {'params': model.dec2.parameters()},
        {'params': model.dec3.parameters()},
        {'params': model.out.parameters(), 'lr': 1e-3},
    ]

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-5)  # L2 regularisation (16_07_2024)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

    # Function to verify the test loader (29/06/2024)
    def verify_test_loader(test_loader):
        print(f"Number of batches in test loader: {len(test_loader)}")
        for idx, (img, mask) in enumerate(test_loader):
            print(f"Test batch {idx}: img shape = {img.shape}, mask shape = {mask.shape}")

    # Check test loader
    # verify_test_loader(test_loader)

    # Load saved model
    if args.load_model == 'True':
        print(f"Load model is {args.load_model}")
        if device == torch.device("cpu"):
            load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), model, optimizer)
        else:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    check_accuracy(val_loader, model, device=device, num_class=args.num_class)
    scaler = torch.cuda.amp.GradScaler()
    # Gradient Accumulation step
    # gradient_accumulations = args.grad_accum
    # print(f"Gradient accumulations = {gradient_accumulations}")

    ############################################# Train ###############################################################
    for epoch in range(args.epochs):
        since = time.time()
        freq_train(train_loader, model, optimizer, criterion, scaler, args.num_class, device, scheduler)
        # train(train_loader, model, optimizer, criterion, scaler, gradient_accumulations, args.num_class, device)

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
        print("Epoch:{}/{}..".format(epoch + 1, args.epochs),
              "Time: {:.2f}m".format((time.time() - since) / 60))

        # print some examples to a folder
        # save_predictions_as_imgs(val_loader, model, num_class=args.num_class, folder="saved_images/", device=device)
        # wandb.save(os.path.join('saved_images', '*'))

        # save_table(val_loader, args.num_class, model,  "Predictions", device)

    ################################################ Test #############################################################
    print("Testing Data...")
    test_accuracy(test_loader, model, device=device, num_class=args.num_class)

    # print some examples to a folder
    # save_predictions_as_imgs(test_loader, model, num_class=args.num_class, folder="saved_images/", device=device)
    wandb.save(os.path.join('saved_images', '*'))

    # save_table(test_loader, args.num_class, model,  "Predictions", device)  # this was commented out
    test_set = test_ds

    mob_miou = miou_score(model, test_set)

    mob_acc = pixel_acc(model, test_set)

    print('Test Set mIoU', np.mean(mob_miou))

    print('Test Set Pixel Accuracy', np.mean(mob_acc))

    i = 0
    # for i in tqdm(range(len(test_set))):
    # for i in range(len(X_test[:2])):
    for i in range(len(X_test)):
        if i % args.saved_images == 0:
            test_set = test_ds
            image, mask = test_set[i]
            filename = get_filename(test_ds, i)
            pred_mask, score = predict_image_mask_miou(model, image, mask, device=device)
            # plot(image.permute(1, 2, 0).detach().cpu()[:, :, 0], mask, pred_mask, score) # green original image
            plot(image, mask, pred_mask, score)
            # plt.savefig('saved_images/prediction' + str(i) + '.jpg')
            plt.savefig(f'saved_images/prediction_{i}_image_{filename}.jpg')
            wandb.save(os.path.join('saved_images', '*'))
            plt.close('all')
        else:
            i += 1

    wandb.save('my_checkpoint.pth.tar')


if __name__ == "__main__":
    main()