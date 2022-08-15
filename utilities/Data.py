# Create a data loading
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


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
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # create some pre-process for the mask, and will look for where it is equal to 255 and will change that to 1
        mask[mask == 255.0] = 1.0  # the reason is because I am using a sigmoid on the last activation indicating the probability it is white pixel and to make sure that it is correct for labels, we will convert those to one.

        # Data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
