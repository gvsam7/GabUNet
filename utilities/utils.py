import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from utilities.Data import Dataset
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


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def get_loaders(
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

    return train_loader, val_loader, test_loader, test_ds


def dataframe(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            # name.append(filename.split('.')[0]  # for <name>.jpg images
            name.append(filename.rsplit('.', 1)[0])  # for <name>.<name>.jpg images

        return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


def database(data):
    if data == "landcover_ai":
        image_path = "Data_LandcoverAI/train_images/"
        mask_path = "Data_LandcoverAI/train_masks/"
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


def plot(image, mask, pred_mask, score):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    # Denormalise images so they will not be dark
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    image = image * std[:, None, None] + mean[:, None, None]

    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('Image')
    ax1.set_axis_off()
    ax1.plot()

    ax2.imshow(mask)
    ax2.set_title('Mask')
    ax2.set_axis_off()
    ax2.plot()

    ax3.imshow(pred_mask)
    ax3.set_title('Predicted Mask | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()
    ax3.plot()


def jaccard(pred_mask, mask, smooth=1e-10, n_classes=5):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
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

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def predict_image_mask_miou(model, image, mask, device='cpu'):
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = jaccard(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def pixel_accuracy(preds, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(preds, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy


def predict_image_mask_pixel(model, image, mask, device='cpu'):
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        preds = model(image)
        acc = pixel_accuracy(preds, mask)
        masked = torch.argmax(preds, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def test_accuracy(loader, model, num_class, device="cuda"):
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
    # return med_jaccard, dice, jaccard, num_correct, num_pixels

    print(f"Got {num_correct}/{num_pixels} pixels with accuracy: {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    print(f"Test Dice score: {dice/len(loader)}")
    print(f"Test IoU score: {jaccard/len(loader)}")
    print(f"Test mIoU score: {med_jaccard/len(loader)}")
    accuracy = num_correct/num_pixels*100
    model.train()
    # wandb.log({"Dice Score": dice_score/len(loader)})
    wandb.log({"Test Dice Score": dice/len(loader)})
    wandb.log({"Test IoU score": jaccard/len(loader)})
    wandb.log({"Test Accuracy": accuracy})
    wandb.log({"Test mIoU Score": med_jaccard/len(loader)})


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
    # return med_jaccard, dice, jaccard, num_correct, num_pixels

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

        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])
        im = im * std[:, None, None] + mean[:, None, None]

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        # plt.imshow(im[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.imshow(im[0].permute(1, 2, 0).cpu())
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

