"""
Author: Georgios Voulgaris
Date: 20/05/2024
Description: UNETR_2D:
            - Architecture: Combines a Vision Transformer (ViT) encoder with a U-Net-style decoder.
            - Encoder: Uses a pure Transformer encoder to process image patches, capturing global context and long-range
                       dependencies.
            - Decoder: Applies de-convolutional layers and concatenates with skip connections from the Transformer
                       encoder.
            - Strengths: Effective at capturing both global and local features due to the Transformer encoder's ability
                         to model long-range dependencies.
"""

import torch
import torch.nn as nn
from utilities.Hyperparameters import arguments
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNETR_2D(nn.Module):
    def __init__(self, in_channels, num_class, cf):
        super().__init__()
        self.cf = cf

        # Patch + Position Embeddings
        self.patch_size = cf["patch_size"]
        self.num_channels = cf["num_channels"]
        self.hidden_dim = cf["hidden_dim"]
        self.patch_embed = nn.Linear(self.patch_size * self.patch_size * self.num_channels, self.hidden_dim)
        # Patch size = 16x16 → (768 // 16) * (768 // 16) = 48 * 48 = 2304 patches
        # Patch size = 512x512 -> (512 // 16) * (512 // 16) = 32 * 32 = 1024 patches
        # Patch size = 32x32 → (768 // 32) * (768 // 32) = 24 * 24 = 576 patches
        self.pos_embed = nn.Embedding(1000, self.hidden_dim)  # Max number of patches set to 1000 for generalisation

        # Transformer Encoder
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=cf["num_heads"],
                dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"],
                activation=nn.GELU(),
                batch_first=True
            ) for _ in range(cf["num_layers"])
        ])

        # CNN Decoder
        self.d1 = DeconvBlock(self.hidden_dim, 512)
        self.s1 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 512),
            ConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512 + 512, 512),
            ConvBlock(512, 512)
        )

        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256 + 256, 256),
            ConvBlock(256, 256)
        )

        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128 + 128, 128),
            ConvBlock(128, 128)
        )

        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ConvBlock(64, 64)
        )

        self.output = nn.Conv2d(64, num_class, kernel_size=1, padding=0)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        patch_height, patch_width = height // self.patch_size, width // self.patch_size
        num_patches = patch_height * patch_width

        # Flatten input image into patches
        patches = inputs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, num_patches, -1)

        # Patch + Position Embeddings
        patch_embed = self.patch_embed(patches)
        pos_embed = self.pos_embed(torch.arange(num_patches, device=inputs.device))
        x = patch_embed + pos_embed

        # Transformer Encoder
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if (i + 1) in skip_connection_index:
                skip_connections.append(x)

        z3, z6, z9, z12 = skip_connections

        # Reshape the skip connection outputs for the decoder
        z0 = inputs
        shape = (batch_size, self.hidden_dim, patch_height, patch_width)
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12 = z12.permute(0, 2, 1).contiguous().view(shape)

        # Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        # Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        # Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)  # Added for patch-size=32, remove for 16
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        # Output
        output = self.output(x)
        return output

"""
###################################################### Test ############################################################
if __name__ == "__main__":
    args = arguments()
    config = {
        "image_size": args.image_size,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "mlp_dim": args.mlp_dim,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "num_patches": (args.image_size // args.patch_size) ** 2,
        "patch_size": args.patch_size,
        "num_channels": args.in_channels
    }

    # Define an input
    x = torch.randn(8, 3, 768, 768)

    model = UNETR_2D(in_channels=3, num_class=2, cf=config)
    y = model(x)
    print(f"y: {y.shape}")"""




"""import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNETR_2D(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # Patch + Position Embeddings
        self.patch_embed = nn.Linear(cf["patch_size"] * cf["patch_size"] * cf["num_channels"], cf["hidden_dim"])
        self.positions = torch.arange(0, cf["num_patches"], dtype=torch.int32)
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"])

        # Transformer Encoder
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"],
                nhead=cf["num_heads"],
                dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"],
                activation=nn.GELU(),
                batch_first=True
            ) for _ in range(cf["num_layers"])
        ])

        # CNN Decoder
        self.d1 = DeconvBlock(cf["hidden_dim"], 512)
        self.s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 512),
            ConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512 + 512, 512),
            ConvBlock(512, 512)
        )

        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256 + 256, 256),
            ConvBlock(256, 256)
        )

        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128 + 128, 128),
            ConvBlock(128, 128)
        )

        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ConvBlock(64, 64)
        )

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # Patch + Position Embeddings
        patch_embed = self.patch_embed(inputs)
        positions = self.positions
        pos_embed = self.pos_embed(positions)
        x = patch_embed + pos_embed

        # Transformer Encoder
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if (i + 1) in skip_connection_index:
                skip_connections.append(x)

        z3, z6, z9, z12 = skip_connections
        batch = inputs.shape[0]
        z0 = inputs.view(batch, self.cf["num_channels"], self.cf["image_size"], self.cf["image_size"])

        shape = (batch, self.cf["hidden_dim"], self.cf["patch_size"], self.cf["patch_size"])
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)

        # Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        # Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        # Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        # Output
        output = self.output(x)
        return output


# Test
if __name__ == "__main__":
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 3

# Define an input
x = torch.randn((
  8,
  config["num_patches"],
  config["patch_size"] * config["patch_size"] * config["num_channels"]
))
# print(x.shape)

model = UNETR_2D(config)
y = model(x)
print(f"y: ", y.shape)"""
