import torch.nn as nn
from models.GaborLayer import GaborConv2d


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
