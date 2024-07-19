import torch
import torch.nn as nn
from models.ConvBlock import BatchNormReLU, DilResBlockMP, DilDecoder, GaborConv2d, DilResBlockAMP
from models.LogGaborLayer import LogGaborConv2d, EnhancedFrequencyLogGaborConv2d, FrequencyLogGaborConv2d,\
    DualDomainLogGaborConv2d, DualDomainAttenLogGabConv2d


class DilGabMPResUNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(DilGabMPResUNet, self).__init__()
        self.num_class = num_class

        # Encoder 1
        # self.conv11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.conv11 = GaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.conv11 = LogGaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.conv11 = FrequencyLogGaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.conv11 = DualDomainLogGaborConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv11 = DualDomainAttenLogGabConv2d(in_channels, 64, kernel_size=3, padding=1)

        # The following Do not learn during training
        # self.conv11 = EnhancedFrequencyLogGaborConv2d(in_channels, 64, kernel_size=3, padding=1, num_scales=3)
        # self.conv11 = EnhancedFrequencyLogGaborConv2d(in_channels, 64, kernel_size=3, num_scales=3)

        self.batchrelu = BatchNormReLU(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(3, 64, kernel_size=1, padding=0)

        # Encoder 2 and 3
        self.res2 = DilResBlockMP(64, 128, stride=2)  # DilResBlockAMP(64, 128, stride=2)
        self.res3 = DilResBlockMP(128, 256, stride=2)  # DilResBlockAMP(128, 256, stride=2)

        # Bridge
        self.res4 = DilResBlockMP(256, 512, stride=2)  # DilResBlockAMP(256, 512, stride=2)

        # Decoder
        self.dec1 = DilDecoder(512, 256)
        self.dec2 = DilDecoder(256, 128)
        self.dec3 = DilDecoder(128, 64)

        # Output
        # self.out = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.out = nn.Conv2d(64, self.num_class, kernel_size=1, padding=0)
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

        # Bridge
        b = self.res4(skip3)

        # Decoder
        d1 = self.dec1(b, skip3)
        d2 = self.dec2(d1, skip2)
        d3 = self.dec3(d2, skip1)

        # Output
        out = self.out(d3)
        # out = self.sigmoid(out)

        return out
