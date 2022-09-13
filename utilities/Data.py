import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

import cv2
from torchvision import transforms as T
import torch


class WaterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # list all the files in the folder
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # L is for gray scale images
        # create some pre-process for the mask, and will look for where it is equal to 255 and will change that to 1
        mask[mask == 255.0] = 1.0  # the reason is because I am using a sigmoid on the last activation indicating the probability it is white pixel and to make sure that it is correct for labels, we will convert those to one.

        # Data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


"""class Dataset(Dataset):

    # def __init__(self, img_path, mask_path, ds, mean, std, transform=None, patch=False):
    def __init__(self, img_path, mask_path, ds, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.ds = ds
        self.transform = transform
        self.patches = patch
        # self.mean = mean
        # self.std = std

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.ds[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.ds[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            # img = Image.fromarray(aug['image'])
            img = aug['image']
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        # img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):

        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches
"""


class Dataset(Dataset):

    def __init__(self, img_path, mask_path, ds, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.ds = ds
        self.transform = transform
        # list all the files in the folder
        self.images = os.listdir(img_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.images[index])
        mask_path = os.path.join(self.mask_path, self.images[index].replace(".jpg", ".png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # L is for gray scale images
        # create some pre-process for the mask, and will look for where it is equal to 255 and will change that to 1
        mask[mask == 255.0] = 1.0  # the reason is because I am using a sigmoid on the last activation indicating the probability it is white pixel and to make sure that it is correct for labels, we will convert those to one.

        # Data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask