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
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
from utilities.Hyperparameters import arguments
from utilities.Networks import networks
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


def main():
    args = arguments()
    wandb.init(entity="predictive-analytics-lab", project="SemSeg", config=args)

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
    # Gradient Accumulation step
    # gradient_accumulations = args.grad_accum
    # print(f"Gradient accumulations = {gradient_accumulations}")

    ############################################# Train ###############################################################
    for epoch in range(args.epochs):
        since = time.time()
        train(train_loader, model, optimizer, criterion, scaler, args.num_class, device)
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
        save_predictions_as_imgs(val_loader, model, num_class=args.num_class, folder="saved_images/", device=device)
        wandb.save(os.path.join('saved_images', '*'))

        save_table(val_loader, args.num_class, model,  "Predictions", device)

    ################################################ Test #############################################################
    print("Testing Data...")
    test_accuracy(test_loader, model, device=device, num_class=args.num_class)

    # print some examples to a folder
    save_predictions_as_imgs(test_loader, model, num_class=args.num_class, folder="saved_images/", device=device)
    wandb.save(os.path.join('saved_images', '*'))

    # save_table(test_loader, args.num_class, model,  "Predictions", device)
    test_set = test_ds

    mob_miou = miou_score(model, test_set)

    mob_acc = pixel_acc(model, test_set)

    i = 0
    # for i in tqdm(range(len(test_set))):
    # for i in range(len(X_test[:2])):
    for i in range(len(X_test)):
        if i % args.saved_images == 0:
            test_set = test_ds
            image, mask = test_set[i]
            pred_mask, score = predict_image_mask_miou(model, image, mask, device=device)
            # plot(image.permute(1, 2, 0).detach().cpu()[:, :, 0], mask, pred_mask, score) # green original image
            plot(image, mask, pred_mask, score)
            plt.savefig('saved_images/prediction' + str(i) + '.jpg')
            wandb.save(os.path.join('saved_images', '*'))
            plt.close('all')
        else:
            i += 1

    print('Test Set mIoU', np.mean(mob_miou))

    print('Test Set Pixel Accuracy', np.mean(mob_acc))

    wandb.save('my_checkpoint.pth.tar')


if __name__ == "__main__":
    main()