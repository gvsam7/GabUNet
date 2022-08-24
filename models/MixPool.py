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
    