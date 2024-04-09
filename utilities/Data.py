import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
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

        # Open the image
        image = Image.open(img_path)

        # Get the image mode
        image_mode = image.mode

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Check if the image mode indicates it's a three-channel image
        if image_mode == "RGB":
            # print("Image is three channels (RGB)")
            # print(f"Image: {image}")
            pass  # No preprocessing needed
        else:
            # print("Image is 1 channel (Gray)")
            # Convert the image to a three-channel format with zeros in other channels
            image_array = np.expand_dims(image_array,
                                         axis=-1)  # Expand dimensions to make it compatible for concatenation
            image_array = np.concatenate((image_array, np.zeros_like(image_array), np.zeros_like(image_array)), axis=-1)
            # Convert the numpy array back to a PIL image
            image = Image.fromarray(image_array.astype(np.uint8))
            # print(f"Modified image: {image}")

        # Load and preprocess the mask
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # Preprocess the mask

        # Apply data augmentation if specified
        if self.transform is not None:
            augmentations = self.transform(image=np.array(image), mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def aname(dataset, index):
    return dataset.images[index]


"""
# This is the original, working for 3-channel RGB code!!!
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
        # print("Image Path:", img_path)
        # print("Mask Path:", mask_path)
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


def get_filename(dataset, index):
    return dataset.images[index]
