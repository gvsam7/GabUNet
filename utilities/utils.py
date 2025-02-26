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
import csv
from torchvision.utils import save_image
import uuid
import json
from torchvision.utils import make_grid


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# 30/06/2024
def save_loader_to_csv(loader, dataset, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Mask'])

        for idx in range(len(dataset)):
            img_path = os.path.join(dataset.img_path, dataset.ds[idx] + '.jpg')
            mask_path = os.path.join(dataset.mask_path, dataset.ds[idx] + '.png')
            writer.writerow([img_path, mask_path])

    print(f'CSV file created: {csv_filename}')


# 30/06/2024
def read_csv(csv_filename):
    data = {'Split': [], 'Image': [], 'Mask': []}
    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for row in reader:
            data['Split'].append(row[0])
            data['Image'].append(row[1])
            data['Mask'].append(row[2])
    return data


# 30/06/2024
def verify_test_loader(test_loader):  # , expected_images, expected_masks):
    test_loader_csv = 'test_loader.csv'
    test_loader_data = pd.read_csv(test_loader_csv)
    expected_images = test_loader_data['Image'].tolist()
    expected_masks = test_loader_data['Mask'].tolist()

    test_loader_images = []
    test_loader_masks = []

    for img, mask in test_loader:
        # Assuming img and mask are torch tensors with filenames as strings
        for i in range(img.size(0)):
            test_loader_images.append(img[i])
            test_loader_masks.append(mask[i])

    test_loader_images = set(test_loader_images)
    test_loader_masks = set(test_loader_masks)

    expected_images_set = set(expected_images)
    expected_masks_set = set(expected_masks)

    missing_images = expected_images_set - test_loader_images
    missing_masks = expected_masks_set - test_loader_masks

    if not missing_images and not missing_masks:
        print("All images and masks processed in test_loader are found in test_loader.csv.")
    else:
        print("Mismatch found:")
        if missing_images:
            print("Missing images:")
            for img in missing_images:
                print(img)
        if missing_masks:
            print("Missing masks:")
            for mask in missing_masks:
                print(mask)

    """# Extract unique image and mask file paths from temp_files (which is test_loader in this case)
    temp_image_files = [file for file in test_loader if file.startswith('original_image')]
    temp_mask_files = [file for file in test_loader if file.startswith('original_mask')]

    processed_images = set(temp_image_files)
    processed_masks = set(temp_mask_files)

    missing_images = [img for img in expected_images if img not in processed_images]
    missing_masks = [mask for mask in expected_masks if mask not in processed_masks]

    if not missing_images and not missing_masks:
        print("All images and masks processed in save_table are found in test_loader.")
    else:
        print("Mismatch found:")
        if missing_images:
            print("Missing images:")
            for img in missing_images:
                print(img)
        if missing_masks:
            print("Missing masks:")
            for mask in missing_masks:
                print(mask)"""


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

    # Save the loaders to CSV files (30/06/2024)
    save_loader_to_csv(train_loader, train_ds, 'train_loader.csv')
    save_loader_to_csv(val_loader, val_ds, 'val_loader.csv')
    save_loader_to_csv(test_loader, test_ds, 'test_loader.csv')

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
    elif data == "WHDLD":
        image_path = "Data_WHDLD/train_images/"
        mask_path = "Data_WHDLD/train_masks/"
    elif data == 'uavid':
        image_path = "Data_UAVid/train_images/"
        mask_path = "Data_UAVid/train_masks/"
    elif data == 'treecrown':
        image_path = "Data_TreeCrown/train_images/"
        mask_path = "Data_TreeCrown/train_masks/"
    elif data == 'treecrowncanada':
        image_path = "Data_TreeCrownCanada/train_images/"
        mask_path = "Data_TreeCrownCanada/train_masks/"
    elif data == "treecrown_ndvi":
        image_path = "Data_TreeCrownNDVI/train_images"
        mask_path = "Data_TreeCrownNDVI/train_masks"
    else:
        image_path = "Data_Water/train_images/"
        mask_path = "Data_Water/train_masks/"
    print(f"Dataset: {data}")
    return image_path, mask_path


def train(loader, model, optimizer, criterion, scaler, num_class, device):
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
            loss = criterion(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        """
        scaler.scale(loss / gradient_accumulations).backward()

        if (batch_idx + 1) % gradient_accumulations == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            """

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

            if true_label.long().sum().item() == 0: # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
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

    # return med_jaccard / len(loader)  # for TransUNet


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


# 30/06/2024
def save_predictions_as_imgs(test_loader, model, num_class, folder="saved_images/", device="cuda"):
    # print(f"Number of batches in loader: {len(test_loader)}")
    model.eval()

    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device=device)

        with torch.no_grad():
            if num_class == 1:
                preds = torch.sigmoid(model(images))
                preds = (preds > 0.5).float()
            else:
                softmax = torch.nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(images)), dim=1).float()

        for idx in range(images.size(0)):
            # Constructing a unique filename based on batch index and image index
            filename = f"prediction_{batch_idx * test_loader.batch_size + idx}.png"

            # Save original image
            img_filename = f"img_{filename}"
            save_image(images[idx], os.path.join(folder, img_filename))

            # Save ground truth mask
            mask_filename = f"mask_{filename}"
            save_image(masks[idx], os.path.join(folder, mask_filename))

            # Save predicted mask (or segmentation)
            pred_filename = f"pred_{filename}"
            save_image(preds[idx], os.path.join(folder, pred_filename))

    model.train()


def save_predictions_as_imgs_Original(test_loader, model, num_class, folder="saved_images/", device="cuda"):
    print(f"Number of batches in loader: {len(test_loader)}")
    model.eval()
    for idx, (img, mask) in enumerate(test_loader):
        print(f"Batch {idx}: img shape = {img.shape}, mask shape = {mask.shape}")
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
        """torchvision.utils.save_image(mask.unsqueeze(1), f"{folder}{idx}.png")
        torchvision.utils.save_image(img, f"{folder}/img_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")"""
        for i in range(img.size(0)):
            image_idx = idx * test_loader.batch_size + i  # Calculate global image index
            img_file = f"{folder}/img_{image_idx}.png"
            mask_file = f"{folder}/mask_{image_idx}.png"
            pred_file = f"{folder}/pred_{image_idx}.png"

            print(f"Saving: {img_file}, {mask_file}, {pred_file}")

            torchvision.utils.save_image(mask[i].unsqueeze(0), mask_file)
            torchvision.utils.save_image(img[i], img_file)
            torchvision.utils.save_image(preds[i], pred_file)
            """
            torchvision.utils.save_image(mask[i].unsqueeze(0), f"{folder}/mask_{image_idx}.png")
            torchvision.utils.save_image(img[i], f"{folder}/img_{image_idx}.png")
            torchvision.utils.save_image(preds[i], f"{folder}/pred_{image_idx}.png")"""

    model.train()


def save_and_logwandb(image_tensor, filename):
    plt.figure(figsize=(10, 10))
    plt.axis("off")

    # Convert to numpy and handle both 2D and 3D tensors
    image_np = image_tensor.cpu().squeeze().numpy()

    # If the image is 3D (C, H, W), convert to (H, W, C)
    if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3, 4]:
        image_np = np.transpose(image_np, (1, 2, 0))

    # Handle grayscale and RGB images
    if len(image_np.shape) == 2 or image_np.shape[-1] == 1:
        plt.imshow(image_np, cmap='gray')
    else:
        plt.imshow(image_np)

    plt.savefig(filename)
    plt.close()
    return filename  # Return the filename instead of wandb.Image object


###################################### Test Test Test 30/06/2024 ##############################################
def save_table(test_loader, num_class, model, save_folder, device):
    print("Saving visual table........")
    model.eval()  # Set model to evaluation mode

    os.makedirs(save_folder, exist_ok=True)

    with torch.no_grad():
        for bx, (im, mask) in enumerate(tqdm(test_loader, total=len(test_loader))):
            im = im.to(device=device)
            mask = mask.to(device=device)

            if num_class == 1:
                pred_mask = torch.sigmoid(model(im))
                pred_mask = (pred_mask > 0.5).float()
            else:
                softmax = nn.Softmax(dim=1)
                pred_mask = torch.argmax(softmax(model(im)), axis=1)

            # Create figure
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Image {bx}', fontsize=16)

            # Display original image
            axs[0].imshow(make_grid(im[0], normalize=True).permute(1, 2, 0).cpu())
            axs[0].set_title('Image')
            axs[0].axis('off')

            # Display original mask
            axs[1].imshow(mask[0].squeeze().cpu(), cmap='gray')
            axs[1].set_title('Mask')
            axs[1].axis('off')

            # Display predicted mask
            axs[2].imshow(pred_mask[0].squeeze().cpu(), cmap='gray')
            axs[2].set_title('Predicted Mask')
            axs[2].axis('off')

            # Save the figure
            unique_id = uuid.uuid4().hex[:6]
            plt.savefig(os.path.join(save_folder, f'segmentation_result_{bx}_{unique_id}.png'))
            plt.close(fig)

    model.train()  # Set model back to training mode

    print(f"Visual table saved in {save_folder}")


def save_and_log(tensor, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_image(tensor, filename)
    return filename


def save_table_data_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Original Image', 'Original Mask', 'Predicted Mask']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def save_table_data_json(data, filename):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)


##################################################################################################################
def save_tablewandb(test_loader, num_class, model, table_name, device):
    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'])
    model.eval()  # Set model to evaluation mode

    temp_files = []  # List to keep track of temporary files

    with torch.no_grad():
        for bx, (im, mask) in enumerate(tqdm(test_loader, total=len(test_loader))):
            im = im.to(device=device)
            mask = mask.to(device=device)

            if num_class == 1:
                pred_mask = torch.sigmoid(model(im))
                pred_mask = (pred_mask > 0.5).float()
            else:
                softmax = nn.Softmax(dim=1)
                pred_mask = torch.argmax(softmax(model(im)), axis=1)

            # Generate unique filenames
            unique_id = uuid.uuid4()
            orig_img_file = f"original_image_{unique_id}.png"
            orig_mask_file = f"original_mask_{unique_id}.png"
            pred_mask_file = f"predicted_mask_{unique_id}.png"

            # Save images and keep track of filenames
            orig_img_file = save_and_log(im[0], orig_img_file)
            orig_mask_file = save_and_log(mask[0], orig_mask_file)
            pred_mask_file = save_and_log(pred_mask[0], pred_mask_file)

            temp_files.extend([orig_img_file, orig_mask_file, pred_mask_file])

            # Add images to the table
            table.add_data(
                wandb.Image(orig_img_file),
                wandb.Image(orig_mask_file),
                wandb.Image(pred_mask_file)
            )

            # Break after processing a few images to keep the table size manageable
            if bx >= 9:  # Adjust this number as needed
                break

    wandb.log({table_name: table})
    model.train()  # Set model back to training mode

    # Clean up temporary files
    for file in temp_files:
        try:
            os.remove(file)
        except OSError:
            pass


############################## Origninal (remove_Original) 30/06/2024 ######################################
def save_table_Original(loader, num_class, model, table_name, device):
    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types=False)

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

        mean = torch.tensor([0.0, 0.0, 0.0], device=device)
        std = torch.tensor([1.0, 1.0, 1.0], device=device)
        im = im * std[:, None, None] + mean[:, None, None]

        # Save images with unique filenames
        original_image_filename = f"original_image_{bx}.jpg"
        original_mask_filename = f"original_mask_{bx}.jpg"
        predicted_mask_filename = f"predicted_mask_{bx}.jpg"

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        # plt.imshow(im[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.imshow(im[0].permute(1, 2, 0).cpu())
        # plt.savefig("original_image.jpg")
        plt.savefig(original_image_filename)
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        # plt.savefig("original_mask.jpg")
        plt.savefig(original_mask_filename)
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(_mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        # plt.savefig("predicted_mask.jpg")
        plt.savefig(predicted_mask_filename)
        plt.close()

        """table.add_data(
            wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
        )"""

        # Add images to Wandb table
        table.add_data(
            wandb.Image(cv2.cvtColor(cv2.imread(original_image_filename), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread(original_mask_filename), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread(predicted_mask_filename), cv2.COLOR_BGR2RGB))
        )

        # Clear temporary images
        for filename in [original_image_filename, original_mask_filename, predicted_mask_filename]:
            try:
                os.remove(filename)
            except OSError:
                pass

        """
        # Save and log images
        def save_and_log(image_tensor, filename):
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(image_tensor.permute(1, 2, 0).cpu())
            plt.savefig(filename)
            plt.close()
            return wandb.Image(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))

        original_image = save_and_log(im[0], "original_image.jpg")
        original_mask = save_and_log(mask.permute(1, 2, 0).detach(), "original_mask.jpg")
        predicted_mask = save_and_log(_mask.permute(1, 2, 0).detach(), "predicted_mask.jpg")

        table.add_data(original_image, original_mask, predicted_mask)

        # Clear temporary images to avoid conflicts
        for filename in ["original_image.jpg", "original_mask.jpg", "predicted_mask.jpg"]:
            try:
                os.remove(filename)
            except OSError:
                pass"""

    wandb.log({table_name: table})

