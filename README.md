# GabUNet

## Description

This project is aiming to research water detection using semantic segmentation.

*Data:* The dataset is comprised of water body images (lakes, rivers) taken from unmanned aerial vehicles (UAV). 
The images were annotated using VGG Image Annotator (VIA). Then, from the .json file masks of the water bodies were created.

*Architecture:* Initially, a UNet is used to measure its performance when trained on the waterbodies dataset. Measurements are in the form of:
- Visual inspection of the produced masks on test data, 
- Per pixel accuracy, 
- Dice Score.

*Aim:* This is going to be a test platform of testing various architectures.  

## Google Colab
- UNet (CAM)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-HAt29Kz1Lj8f8AnThhzK0s4IQi7q3sN)


## Papers
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

