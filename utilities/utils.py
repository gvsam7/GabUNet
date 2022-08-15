import torch
import torchvision
from utilities.Data import WaterDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import wandb


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = WaterDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = WaterDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    accuracy = num_correct/num_pixels*100
    model.train()
    wandb.log({"Dice Score": dice_score/len(loader)})
    wandb.log({"Accuracy": accuracy})


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


# def save_table(loader, model, table_name, device):
#     table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types=True)
#
#     for bx, data in tqdm(enumerate(loader), total=len(loader)):
#         im, mask = data
#         im = im.to(device=device)
#         mask = mask.to(device=device)
#         _mask = model(im)
#         _, _mask = torch.max(_mask, dim=1)
#
#         plt.figure(figsize=(10, 10))
#         plt.axis("off")
#         plt.imshow(im[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
#         plt.savefig("original_image.jpg")
#         plt.close()
#
#         plt.figure(figsize=(10, 10))
#         plt.axis("off")
#         plt.imshow(mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
#         plt.savefig("original_mask.jpg")
#         plt.close()
#
#         plt.figure(figsize=(10, 10))
#         plt.axis("off")
#         plt.imshow(_mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
#         plt.savefig("predicted_mask.jpg")
#         plt.close()
#
#         table.add_data(
#             wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
#             wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
#             wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
#         )
#
#     wandb.log({table_name: table})


def save_table(loader, model, table_name, device, folder="saved_images/"):
    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types=True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

        print(type(y), y.shape)
        y1 = y.unsqueeze(1)
        print(type(y1), y1.shape)
        print(type(preds), preds.shape)
        pred1 = pred1.squeeze(1)
        print(type(pred1), pred1.shape)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(x[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_image.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(y.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_mask.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        # plt.imshow(preds.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.imshow(preds.permute(2, 3, 0, 1).detach().cpu()[:, :, :, 0])
        plt.savefig("predicted_mask.jpg")
        plt.close()

        table.add_data(
            wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
        )

    wandb.log({table_name: table})

    model.train()