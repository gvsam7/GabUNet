import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from utilities.utils import num_parameters
from models.LogGaborLayer import LogGaborConv2d
from models.MixPool import MixPool
from models.ConvBlock import DACBlock


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # instead of inplace=True
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)  # commented for dilated conv
        # Dilated convolution
        self.dil = DACBlock(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)  # commented for dilated conv
        # added dilated conv
        out = self.dil(out)

        out = out + identity
        out = self.relu(out)
        return out


class CustomResNet18(nn.Module):
    def __init__(self, in_channels, num_classes=None):
        super(CustomResNet18, self).__init__()

        self.in_channels = 64
        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1 = LogGaborConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification head (optional for feature extraction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if num_classes is not None:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = None

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                # nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # commented for mixpooling
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                MixPool(2, 2, 0, 0.8),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)  # 1/4 resolution
        feat2 = self.layer2(feat1)  # 1/8 resolution
        feat3 = self.layer3(feat2)  # 1/16 resolution
        feat4 = self.layer4(feat3)  # 1/32 resolution

        if self.fc is not None:
            x = self.avgpool(feat4)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            return feat1, feat2, feat3, feat4  # For DeepLabV3+

        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        # Parallel branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in atrous_rates
        ])

        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Projection layer to merge branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1x1 convolution
        feat1 = self.conv1x1(x)

        # Atrous convolutions
        atrous_feats = [conv(x) for conv in self.atrous_convs]

        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Combine all features
        feats = [feat1] + atrous_feats + [global_feat]
        feats = torch.cat(feats, dim=1)

        # Project to final feature space
        return self.project(feats)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = CustomResNet18(in_channels=in_channels)  # Existing backbone

        # ASPP module
        self.aspp = ASPP(
            in_channels=512,  # Final ResNet stage channels
            out_channels=256,
            atrous_rates=(6, 12, 18)
        )

        # Low-level feature projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        # Backbone forward pass
        low_level_feat, _, _, high_level_feat = self.backbone(x)

        # ASPP on high-level features
        aspp_feat = self.aspp(high_level_feat)

        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Upsample ASPP features
        aspp_feat = F.interpolate(aspp_feat, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate features
        concat_feat = torch.cat([aspp_feat, low_level_feat], dim=1)

        # Decoder
        decoder_feat = self.decoder(concat_feat)

        # Final classification
        output = self.classifier(decoder_feat)

        # Upsample to original input size
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)

        return output


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
# Testing dummy values
def main():
    # Test the model with dummy input
    dummy_input = torch.randn(1, 3, 768, 768)
    model = DeepLabV3Plus(in_channels=3, num_classes=5)
    model.eval()  # Set the model to evaluation mode
    print(model)
    n_parameters = num_parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")
    output = model(dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()"""