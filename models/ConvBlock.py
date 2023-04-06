import torch.nn as nn
import torch
from models.GaborLayer import GaborConv2d
from models.MixPool import MixPool


class DACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DACBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=3)
        self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=6)
        self.conv9 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=9)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x6 = self.conv6(x)
        x9 = self.conv9(x)
        x_t = x1 + (x3 + x6 + x9)/3
        return self.relu(self.bn(x_t))


class BatchNormReLU(nn.Module):
    def __init__(self, in_channels):
        super(BatchNormReLU, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # Convolutional layer
        self.bn1 = BatchNormReLU(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = BatchNormReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        # Shortcut Connection (Identity Mapping)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        s = self.skip(inputs)

        skip = x + s
        return skip


class ResBlockMP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockMP, self).__init__()

        # Convolutional layer
        self.bn1 = BatchNormReLU(in_channels)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                MixPool(2, 2, 0, 0.8)
            )
        self.bn2 = BatchNormReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        # Shortcut Connection (Identity Mapping)
        self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
                MixPool(2, 2, 0, 0.8)
            )

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        s = self.skip(inputs)

        skip = x + s
        return skip


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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Max pooling
class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Added MaxPool to the original
        )

    def forward(self, x):
        return self.conv(x)


class DilConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DACBlock(out_channels, out_channels)
            # nn.MaxPool2d(kernel_size=2, stride=2) # Added MaxPool to the original
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DilBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilBottleneck, self).__init__()
        self.conv1 = DilACBlock(in_channels, out_channels)
        self.conv2 = DilACBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GConvBlock, self).__init__()
        self.conv = nn.Sequential(
            GaborConv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # n.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.avg_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False),

        )
        self.max_out = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.avg_out(x)
        max_out = self.max_out(x)
        out = avg_out + max_out
        del avg_out, max_out
        return x * self.sigmoid(out)


class ACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ACBlock, self).__init__()
        self.square = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.square(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.relu(self.bn(x1 + x2 + x3))


class GACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(GACBlock, self).__init__()
        self.square = GaborConv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.square(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.relu(self.bn(x1 + x2 + x3))


class DACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DACBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=3)
        self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=6)
        self.conv9 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=9)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x6 = self.conv6(x)
        x9 = self.conv9(x)
        x_t = x1 + (x3 + x6 + x9)/3
        return self.relu(self.bn(x_t))


class DilChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=2):
        super(DilChannelAttention, self).__init__()
        self.conv = DilACBlock(in_planes, out_planes, 1, bias=False)
        self.avg_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False),

        )
        self.max_out = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.avg_out(x)
        max_out = self.max_out(x)
        out = avg_out + max_out
        del avg_out, max_out
        return x * self.sigmoid(out)


class DilACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilACBlock, self).__init__()
        self.square1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1)
        self.cross_ver1 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding='same', stride=1)
        self.cross_hor1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding='same', stride=1)
        self.square3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=2)
        self.cross_ver3 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding='same', stride=1, dilation=2)
        self.cross_hor3 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding='same', stride=1, dilation=2)
        self.square6 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=6)
        self.cross_ver6 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding='same', stride=1, dilation=6)
        self.cross_hor6 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding='same', stride=1, dilation=6)
        self.square9 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=9)
        self.cross_ver9 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding='same', stride=1, dilation=9)
        self.cross_hor9 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding='same', stride=1, dilation=9)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x11 = self.square1(x)
        x12 = self.cross_ver1(x)
        x13 = self.cross_hor1(x)
        x1 = x11 + x12 + x13
        x31 = self.square3(x)
        x32 = self.cross_ver3(x)
        x33 = self.cross_hor3(x)
        x3 = x31 + x32 + x33
        x61 = self.square6(x)
        x62 = self.cross_ver6(x)
        x63 = self.cross_hor6(x)
        x6 = x61 + x62 + x63
        x91 = self.square9(x)
        x92 = self.cross_ver9(x)
        x93 = self.cross_hor9(x)
        x9 = x91 + x92 + x93
        x = x1 + (x3 + x6 + x9)/3
        return self.relu(self.bn(x))