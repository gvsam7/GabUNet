import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.Data import WaterDataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import wandb


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    # train_dir,
    # train_maskdir,
    # val_dir,
    # val_maskdir,
    image_path,
    mask_path,
    X_train,
    X_val,
    X_test,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    """train_ds = WaterDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )"""

    train_ds = Dataset(
        img_path=image_path,
        mask_path=mask_path,
        ds=X_train,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    """val_ds = WaterDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )"""
    val_ds = Dataset(
        img_path=image_path,
        mask_path=mask_path,
        ds=X_val,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = Dataset(
        img_path=image_path,
        mask_path=mask_path,
        ds=X_test,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def check_accuracy(loader, model, num_class, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    # IoU = 0
    med_jaccard = 0
    jaccard = 0
    dice = 0
    model.eval()

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            # mask = mask.to(device).unsqueeze(1)
            if num_class == 1:
                mask = mask.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(img))  # when binary 1-class semantic segmentation
                preds = (preds > 0.5).float()  # 1 class semantic segmentation
            else:
                mask = mask.to(device)
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(img)), axis=1)
                # preds = F.softmax((model(img)), dim=1)

            num_correct += (preds == mask).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * mask).sum()) / ((preds + mask).sum() + 1e-8)
            # IoU += (preds * mask).sum() / ((preds + mask).sum() + 1e-8)

            med_jaccard += mIoU(preds, mask, num_class)
            dice += Dice(preds, mask, num_class)
            jaccard += IoU(preds, mask, num_class)

    print(f"Got {num_correct}/{num_pixels} pixels with accuracy: {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")
    print(f"IoU score: {jaccard/len(loader)}")
    print(f"mIoU score: {med_jaccard/len(loader)}")
    accuracy = num_correct/num_pixels*100
    model.train()
    # wandb.log({"Dice Score": dice_score/len(loader)})
    wandb.log({"Dice Score": dice/len(loader)})
    wandb.log({"IoU score": jaccard/len(loader)})
    wandb.log({"Accuracy": accuracy})
    wandb.log({"mIoU Score": med_jaccard/len(loader)})


def IoU(pred_mask, mask, n_classes, smooth=1e-10):
    with torch.no_grad():
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        jaccard_per_class = []
        for clas in range(0, n_classes):
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # Non existing labels in this loop
                jaccard_per_class.append(np.nan)

            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                jaccard = intersect / (union + smooth)
                jaccard_per_class.append(jaccard)
        return np.nanmean(jaccard_per_class)


def Dice(pred_mask, mask, n_classes, smooth=1e-10):
    with torch.no_grad():
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        dice_per_class = []
        for clas in range(0, n_classes):
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # Non existing labels in this loop
                dice_per_class.append(np.nan)

            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                dice = (2 * intersect) / (union + smooth)
                dice_per_class.append(dice)
        return np.nanmean(dice_per_class)


def mIoU(pred_mask, mask, n_classes, smooth=1e-10):
    with torch.no_grad():
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def save_predictions_as_imgs(loader, model, num_class, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (img, mask) in enumerate(loader):
        img = img.to(device=device)
        with torch.no_grad():
            if num_class == 1:
                preds = torch.sigmoid(model(img))
                preds = (preds > 0.5).float()
            else:
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(img)), axis=1)
                preds = preds.float()
                preds = preds.unsqueeze(1)
        torchvision.utils.save_image(mask.unsqueeze(1), f"{folder}{idx}.png")
        torchvision.utils.save_image(img, f"{folder}/img_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

    model.train()


def save_table(loader, num_class, model, table_name, device):
    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types=True)

    for bx, data in tqdm(enumerate(loader), total=len(loader)):
        im, mask = data
        im = im.to(device=device)
        mask = mask.to(device=device)
        if num_class == 1:
            _mask = torch.sigmoid(model(im))
            _mask = (_mask > 0.5).float()
            _mask = _mask.squeeze(1)
        else:
            softmax = nn.Softmax(dim=1)
            _mask = torch.argmax(softmax(model(im)), axis=1)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(im[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_image.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_mask.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(_mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("predicted_mask.jpg")
        plt.close()

        table.add_data(
            wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
        )

    wandb.log({table_name: table})

