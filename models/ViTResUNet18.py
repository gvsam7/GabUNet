import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Connection (Identity Mapping)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)
        self.res = ResBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        current_size_h, current_size_w = x.size()[2:]
        scale_factor_h = self.output_size[0] / current_size_h
        scale_factor_w = self.output_size[1] / current_size_w

        x = nn.functional.interpolate(x, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=True)

        if skip.size()[2:] != x.size()[2:]:
            skip = nn.functional.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNetEncoder, self).__init__()
        # Load pre-trained ResNet18 model
        resnet = resnet18(pretrained=True)
        # Remove the classification layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # Change the first convolutional layer to accept the desired number of input channels
        self.resnet[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # Assuming x is the input tensor passed to ResNetEncoder
        x = self.resnet(x)
        return x


class ViTResUNet18(nn.Module):
    def __init__(self, in_channels, num_classes, vit_patch_size=4):
        super(ViTResUNet18, self).__init__()
        self.num_classes = num_classes
        self.vit_patch_size = vit_patch_size

        # Encoder
        self.resnet_encoder = ResNetEncoder(in_channels)
        self.patch_embed = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=4
        )

        # Decoder
        self.dec1 = Decoder(512, 256, output_size=(64, 128))
        self.dec2 = Decoder(256, 128, output_size=(128, 256))
        self.dec3 = Decoder(128, 64, output_size=(256, 512))
        self.dec4 = Decoder(64, 32, output_size=(512, 1024))

        # Output
        self.out = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.resnet_encoder(x)
        print(f"Shape x1: ", x1.shape)
        x_patch = self.patch_embed(x1)

        # Calculate the number of patches dynamically
        num_patches_h = x_patch.size(2)
        num_patches_w = x_patch.size(3)

        x_patch_flat = x_patch.flatten(2).transpose(1, 2)  # Flatten and transpose

        # Print the shapes of relevant tensors before reshaping
        print("Shape of x_patch_flat:", x_patch_flat.shape)
        print("num_patches_h:", num_patches_h)
        print("num_patches_w:", num_patches_w)

        # Create positional embedding from flattened patch embeddings
        positional_embedding_flat = nn.Parameter(torch.zeros_like(x_patch_flat))

        pos_encoding_h = torch.linspace(-1.0, 1.0, num_patches_h, device=x_patch_flat.device)
        pos_encoding_w = torch.linspace(-1.0, 1.0, num_patches_w, device=x_patch_flat.device)
        pos_encoding = torch.stack(torch.meshgrid(pos_encoding_h, pos_encoding_w, indexing='ij'), dim=-1).flatten(0, 1)

        # Reshape positional_embedding_flat to match positional_embedding dimensions
        positional_embedding_flat = positional_embedding_flat.view(1, x_patch_flat.size(1), num_patches_h,
                                                                   num_patches_w)

        # Print the shapes after adding positional embeddings
        print("Shape of positional_embedding_flat after adding embeddings:", positional_embedding_flat.shape)

        # Add learnable class token
        class_token = nn.Parameter(torch.zeros(1, 1, x_patch_flat.size(-1), device=x_patch.device))
        x_patch = torch.cat([class_token.expand(x_patch_flat.size(0), -1, -1), x_patch_flat], dim=1)
        x_patch = x_patch.permute(1, 0, 2)  # Permute to (num_patches + 1, batch_size, embedding_dim)
        x_patch = self.transformer_encoder(x_patch)
        x_patch = x_patch.permute(1, 2, 0)  # Permute back to (batch_size, embedding_dim, num_patches + 1)
        x_patch = x_patch[:, :, 1:].reshape(x_patch.size(0), x_patch.size(1), num_patches_h, num_patches_w)

        # Concatenate features from the ResNet encoder
        x_patch = x_patch.permute(0, 1, 2, 3)  # B, C,H, W
        x = torch.cat([x1, x_patch], dim=1)

        # Decoder blocks
        print(f"Decoder 1: ")
        d1 = self.dec1(x, x1)
        print(f"Shape D1: ", d1.shape)
        print(f"Decoder 2: ")
        d2 = self.dec2(d1, x1)
        print(f"Shape D2: ", d2.shape)
        print(f"Decoder 3: ")
        d3 = self.dec3(d2, x_patch)
        print(f"Shape d3: ", d3.shape)
        d4 = self.dec4(d3, x_patch)

        # Output layer
        out = self.out(d4)

        return out

# Test the model with dummy input
if __name__ == "__main__":
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 256, 512)  # Assuming input image size is 256x512 and has 3 channels
    # Create an instance of the ViTResUNet18 model
    model = ViTResUNet18(in_channels=3, num_classes=2)
    # Forward pass through the model
    output = model(dummy_input)
    # Print the shape of the output tensor
    print("Output shape:", output.shape)
