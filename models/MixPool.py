import torch
import torch.nn as nn
import torch.nn.functional as F


class MixPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, alpha):
        super(MixPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (1 - self.alpha) * \
            F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x


class AdaptiveMixPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, in_channels):
        super(AdaptiveMixPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define layers for gating mechanism
        self.gate_layer = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute gating value (alpha) from input x
        alpha = self.sigmoid(self.gate_layer(x))

        # Perform mixed pooling
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

        # Ensure alpha has the same size as the pooled outputs
        alpha = F.interpolate(alpha, size=max_pool.shape[2:], mode='nearest')

        # Use broadcasting to apply alpha
        mixed_pool = alpha * max_pool + (1 - alpha) * avg_pool

        return mixed_pool