import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ConvBlock import GaborConv2d, BatchNormReLU


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res = ResBlock(in_channels+out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], axis=1)
        x = self.res(x)
        return x


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_class):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18():
    return ResNet18(ResBlock, [2, 2, 2, 2])


class ResUNet18(nn.Module):
    def __init__(self, in_channels, num_class):
        super(ResUNet18, self).__init__()
        self.num_class = num_class
        self.in_channels = 64

        # Custom ResNet18 Backbone
        # Encoder 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Encoders 2 - 4
        self.encoder2 = self._make_layer(ResBlock, 64, 2)
        self.encoder3 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.encoder4 = self._make_layer(ResBlock, 256, 2, stride=2)

        # Bridge
        self.encoder5 = self._make_layer(ResBlock, 512, 2, stride=2)

        # Decoder
        self.dec1 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec3 = Decoder(128, 64)
        self.dec4 = Decoder(64, 64)

        # Output
        self.out = nn.Conv2d(64, self.num_class, kernel_size=1, padding=0)

        # Upsample skip1
        self.upsample_skip1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Upsample final output
        self.upsample_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Adjust scale_factor

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, input):
        # Encoder
        skip1 = self.encoder1(input)
        skip1_upsampled = self.upsample_skip1(skip1)

        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        bridge = self.encoder5(skip4)

        # Decoder
        d1 = self.dec1(bridge, skip4)
        d2 = self.dec2(d1, skip3)
        d3 = self.dec3(d2, skip2)
        # d4 = self.dec4(d3, skip1)
        d4 = self.dec4(d3, skip1_upsampled)

        # Output
        out = self.out(d4)
        out = self.upsample_final(out)  # Upsample to match the input size

        return out



"""class ResUNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(ResUNet, self).__init__()
        self.num_class = num_class

        # Encoder 1
        self.conv11 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        # self.conv11 = GaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.batchrelu = BatchNormReLU(32)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(3, 32, kernel_size=1, padding=0)

        # Encoder 2 and 3
        self.res2 = ResBlock(32, 64, stride=2)
        self.res3 = ResBlock(64, 128, stride=2)
        self.res4 = ResBlock(128, 256, stride=2)

        # Bridge
        self.res5 = ResBlock(256, 512, stride=2)

        # Decoder
        self.dec1 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec3 = Decoder(128, 64)
        self.dec4 = Decoder(64, 32)

        # Output
        # self.out = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.out = nn.Conv2d(32, self.num_class, kernel_size=1, padding=0)
        # self.sigmoid = nn.Sigmoid() # if there is only one class

    def forward(self, input):
        # Encoder 1
        x = self.conv11(input)
        x = self.batchrelu(x)
        x = self.conv12(x)
        s = self.conv13(input)
        skip1 = x + s

        # Encoder 2 and 3
        skip2 = self.res2(skip1)
        skip3 = self.res3(skip2)
        skip4 = self.res4(skip3)

        # Bridge
        b = self.res4(skip4)

        # Decoder
        d1 = self.dec1(b, skip4)
        d2 = self.dec2(d1, skip3)
        d3 = self.dec3(d2, skip2)
        d4 = self.dec4(d3, skip1)

        # Output
        out = self.out(d4)
        # out = self.sigmoid(out)

        return out"""
