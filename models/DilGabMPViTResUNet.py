import torch
import torch.nn as nn
from models.ConvBlock import GaborConv2d, DACBlock
from models.MixPool import MixPool
from models.LogGaborLayer import LogGaborConv2d
from utilities.utils import num_parameters

"""
Description: The DilGabMPViTResUNet model is a fusion of convolutional neural network (CNN), transformer architectures,
            and texture bias with a wider receptive field, designed for the task of semantic segmentation. 
            It leverages the strengths of both architectures, plus texture biases to capture spatial information 
            efficiently while also modeling long-range dependencies in the input data. This is achieved by combining the
            strengths of U-Net, CNN, and Vision Transformer (ViT), leveraging the powerful feature extraction 
            capabilities of the ResNet-based encoder, the attention of averaged dilated convolutions, texture biases, 
            and the attention mechanisms of ViT for semantic segmentation tasks. Integrating ViT's attention mechanism 
            can potentially enhance the model's ability to capture long-range dependencies and improve performance.
"""


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        # Initialize the upsample module without specifying scale_factor here
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)
        self.res = ResBlock(in_channels+out_channels, out_channels)

    def forward(self, x, skip):
        # Calculate the scale factor based on the current size of x and the desired output size
        current_size_h, current_size_w = x.size()[2:]
        scale_factor_h = self.output_size[0] / current_size_h
        scale_factor_w = self.output_size[1] / current_size_w

        # Use the calculated scale factor for upsampling
        # Pass scale_factor as a tuple directly to the upsample call
        x = nn.functional.interpolate(x, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=True)

        # Ensure that the skip connection has the same spatial dimensions as x
        if skip.size()[2:] != x.size()[2:]:
            # Resize skip to match x's size if necessary
            skip = nn.functional.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # Convolutional layers
        if stride == 2:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                MixPool(2, 2, 0, 0.8)
            )
            # self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
                MixPool(2, 2, 0, 0.8)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Dilated convolution
        self.dil = DACBlock(out_channels, out_channels)

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # added dilated conv
        x = self.dil(x)

        # x += identity
        x = x + identity
        x = nn.ReLU(inplace=True)(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, in_channels):
        super(ResNetEncoder, self).__init__()
        self.in_channels = 64
        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = GaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv11 = LogGaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mixpool = MixPool(2, 2, 0, 0.8)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Assuming x is the input tensor passed to ResNetEncoder
        # print("Size of input tensor x:", x.size())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.mixpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x


class DilGabMPViTResUNet(nn.Module):
    def __init__(self, in_channels, num_classes, vit_patch_size):
        super(DilGabMPViTResUNet, self).__init__()
        self.num_classes = num_classes
        self.vit_patch_size = vit_patch_size

        # Encoder
        self.resnet_encoder = ResNetEncoder(ResBlock, [2, 2, 2, 2], in_channels=in_channels)
        # self.patch_embed = nn.Conv2d(512, 768, kernel_size=vit_patch_size, stride=vit_patch_size)
        self.patch_embed = nn.Conv2d(256, 512, kernel_size=vit_patch_size, stride=vit_patch_size)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=4
        )

        # Bridge
        # self.bridge = nn.Conv2d(512, 512, kernel_size=1)
        self.bridge = ResBlock(512, 512, stride=2)

        # Decoder
        self.dec1 = Decoder(512, 256, output_size=(256, 512))
        # self.dec2 = Decoder(in_channels=256 + num_skip_channels, out_channels=128, output_size=(128, 256))
        self.dec2 = Decoder(256+128, 128, output_size=(128, 256))
        self.dec3 = Decoder(128+192, 64, output_size=(768, 768))
        # Output
        self.out = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.resnet_encoder(x)
        x_patch = self.patch_embed(x1)

        # Calculate the number of patches dynamically
        img_size = x1.size(-1)
        num_patches_h = x_patch.size(2)
        num_patches_w = x_patch.size(3)
        num_patches = num_patches_h * num_patches_w

        x_patch_flat = x_patch.flatten(2).transpose(1, 2)

        # Transformer Encoder
        x_patch_flat = self.transformer_encoder(x_patch_flat)

        # Reshape to original spatial dimensions
        x_patch = x_patch_flat.transpose(1, 2).view(-1, 512, num_patches_h, num_patches_w)

        # Bridge
        bridge_out = self.bridge(x_patch)

        # Decoder
        d1 = self.dec1(bridge_out, x1)
        d2 = self.dec2(d1, x1)
        d3 = self.dec3(d2, x1)

        # Output
        out = self.out(d3)

        return out

"""
######################################## Test the model with dummy input ###############################################
if __name__ == "__main__":
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 128, 128)  # Assuming input image size is 256x512 and has 3 channels
    # Create an instance of the ViTResUNet18 model
    model = DilGabMPViTResUNet(in_channels=3, num_classes=2, vit_patch_size=1)
    print(model)
    n_parameters = num_parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")
    # Forward pass through the model
    output = model(dummy_input)
    # Print the shape of the output tensor
    print("Output shape:", output.shape)"""