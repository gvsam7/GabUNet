"""
Author: Georgios Voulgaris
Date: 10/07/2024
Description: This is a Log Gabor filter implementation in a convolutional layer. The Log Gabor filter aims to model
             human visual perception by logarithmically scaling the frequency response. Compared to the Original Gabor
             filter, the Log-Gabor filter does not have the same DC problem.
             Formula:
             G(f) = exp\left(-\frac{(\log(f/f_{0}))^2}{2(\log(\sigma/f_{0}))^2}\right)

             Note: that a 2D expansion is added by adding another dimension, hence, the filter is not only designed for
             a particular frequency, but also is designed for a particular orientation. The orientation component is a
             Gaussian distance function according to the angle in polar coordinates.
"""


import torch
from torch import nn
from torch.nn import Parameter
import math
import torch.nn.functional as F


class LogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode="zeros"):
        super(LogGaborConv2d, self).__init__()
        self.is_calculated = False

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              groups,
                              bias,
                              padding_mode
                              )

        self.kernel_size = self.conv.kernel_size

        # Prevents dividing by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini, A. Aghagolzadeh and H. Seyedarabi, "Face recognition
        # using Gabor filter bank, kernel principal component analysis and support vector machine"
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True
        )

        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True
        )

        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(math.pi * torch.rand(out_channels, in_channels), requires_grad=True)

        # Modified: Create f0 and theta0 as learnable parameters for all channels
        # self.f0 = Parameter(torch.ones(out_channels, in_channels), requires_grad=True)
        # self.theta0 = Parameter(torch.ones(out_channels, in_channels), requires_grad=True)
        # Global
        self.f0 = Parameter(torch.Tensor([1.0]), requires_grad=True)  # Define f0 as a parameter
        self.theta0 = Parameter(torch.Tensor([1.0]), requires_grad=True)  # Define theta0 as a parameter

        # Initialise grid parameters
        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0],
            requires_grad=False
        )

        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0],
            requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1])
            ]
        )
        self.x = Parameter(self.x)
        self.y = Parameter(self.y)

        # Initialise filter weights
        self.weight = Parameter(
            torch.empty(self.conv.weight.shape, requires_grad=True),
            requires_grad=True
        )

        # Register all parameters
        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("f0", self.f0)  # Register f0 as a parameter
        self.register_parameter("theta0", self.theta0)  # Register f0 as a parameter
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv.out_channels):
            for j in range(self.conv.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)
                f0 = self.f0.expand_as(self.y)  # global values
                theta0 = self.theta0.expand_as(self.y)  # global values
                # f0 = self.f0[i, j].expand_as(self.y)
                # theta0 = self.theta0[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                r = torch.sqrt(rotx ** 2 + roty ** 2 + self.delta)
                # Log-Gabor filter (radial component)
                g_radial = torch.exp(-1 * ((torch.log(r) - torch.log(f0)) / (2 * torch.log(sigma / f0))) ** 2)

                # Angular component
                g_angular = torch.exp(-((theta - theta0) ** 2) / (2 * sigma ** 2))

                # Combine radial and angular components
                g = g_radial * g_angular

                g = g * torch.cos(freq * r + psi)
                g = g / (2 * math.pi * (sigma ** 2))

                self.conv.weight.data[i, j] = g


class FrequencyLogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(FrequencyLogGaborConv2d, self).__init__()
        self.freq_log_gabor = LogGaborConv2d(in_channels * 2, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        # Frequency Domain Processing
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        x_freq = torch.cat([magnitude, phase], dim=1)
        x_freq = self.freq_log_gabor(x_freq)

        return x_freq


class DualDomainLogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(DualDomainLogGaborConv2d, self).__init__()
        self.freq_log_gabor = LogGaborConv2d(in_channels * 2, out_channels, kernel_size, **kwargs)
        self.spatial_log_gabor = LogGaborConv2d(in_channels, out_channels, kernel_size, **kwargs)

        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Frequency Domain Processing
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        x_freq = torch.cat([magnitude, phase], dim=1)
        x_freq = self.freq_log_gabor(x_freq)

        # Spatial Domain Processing
        x_spatial = self.spatial_log_gabor(x)

        # Combine extracted features from both domains
        x_combined = torch.cat([x_freq, x_spatial], dim=1)
        x = self.fusion(x_combined)
        return x


class FCChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):  # Lower Reduction Ratios: Preserve more information, which can be advantageous for complex tasks but at the cost of increased computational load.
        super(FCChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        # self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.avg_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.max_out = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.conv(x)
        avg_out = self.avg_out(x)
        max_out = self.max_out(x)
        out = avg_out + max_out
        # del avg_out, max_out
        return x * self.sigmoid(out)


class DualDomainAttenLogGabConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, freq_kernel_size=768, **kwargs):  # freq_kernel_size=256
        super(DualDomainAttenLogGabConv2d, self).__init__()

        # Use a larger kernel for frequency domain
        self.freq_log_gabor = LogGaborConv2d(in_channels * 2, out_channels,
                                             kernel_size=(freq_kernel_size, freq_kernel_size), **kwargs)

        # Keep the spatial domain as before
        self.spatial_log_gabor = LogGaborConv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = ChannelAttention(out_channels)

    def forward(self, x):
        # Frequency Domain Processing
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        x_freq = torch.cat([magnitude, phase], dim=1)
        x_freq = self.freq_log_gabor(x_freq)

        # Spatial Domain Processing
        x_spatial = self.spatial_log_gabor(x)

        # Resize frequency domain output to match spatial domain output
        x_freq = F.interpolate(x_freq, size=x_spatial.shape[2:], mode='bilinear', align_corners=False)

        # Combine extracted features from both domains
        x_combined = torch.cat([x_freq, x_spatial], dim=1)
        x = self.fusion(x_combined)
        attention = self.attention(x)
        return attention


# This is working with small kernel_size which is the same for both spatial and frequency domain. However, this small
# kernel_size in the frequency domain does not capture the global information available in the frequency domain, hence
# does not harness the benefit of extracting features in the frequency domain.
class EqualKernelDualDomainAttenLogGabConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(EqualKernelDualDomainAttenLogGabConv2d, self).__init__()
        self.freq_log_gabor = LogGaborConv2d(in_channels * 2, out_channels, kernel_size, **kwargs)
        self.spatial_log_gabor = LogGaborConv2d(in_channels, out_channels, kernel_size, **kwargs)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = ChannelAttention(out_channels)

    def forward(self, x):
        # Frequency Domain Processing
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        x_freq = torch.cat([magnitude, phase], dim=1)
        x_freq = self.freq_log_gabor(x_freq)

        # Spatial Domain Prcessing
        x_spatial = self.spatial_log_gabor(x)

        # Combine extracted features from both domains
        x_combine = torch.cat([x_freq, x_spatial], dim=1)
        x = self.fusion(x_combine)
        attention = self.attention(x)
        return attention


"""
class FrequencyLogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", use_frequency_domain=True):
        super(FrequencyLogGaborConv2d, self).__init__()
        self.use_frequency_domain = use_frequency_domain
        self.is_calculated = False

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode)
        self.kernel_size = self.conv.kernel_size
        self.delta = 1e-3

        # Parameters for Log Gabor filter
        self.freq = nn.Parameter(torch.rand(out_channels, in_channels) * math.pi)
        self.theta = nn.Parameter(torch.rand(out_channels, in_channels) * math.pi)
        self.sigma = nn.Parameter(torch.rand(out_channels, in_channels) * math.pi)
        self.psi = nn.Parameter(torch.rand(out_channels, in_channels) * math.pi)
        self.f0 = nn.Parameter(torch.ones(out_channels, in_channels))
        self.theta0 = nn.Parameter(torch.ones(out_channels, in_channels))

        # Initialise grid parameters
        self.x0 = nn.Parameter(torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False)
        self.y0 = nn.Parameter(torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False)

        y, x = torch.meshgrid([
            torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
            torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1])
        ])
        self.x = nn.Parameter(x, requires_grad=False)
        self.y = nn.Parameter(y, requires_grad=False)

        self.weight = nn.Parameter(torch.empty(self.conv.weight.shape), requires_grad=True)

    def calculate_weights(self):
        for i in range(self.conv.out_channels):
            for j in range(self.conv.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)
                f0 = self.f0[i, j].expand_as(self.y)
                theta0 = self.theta0[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                r = torch.sqrt(rotx ** 2 + roty ** 2 + self.delta)

                # Log-Gabor filter (radial component)
                g_radial = torch.exp(-1 * ((torch.log(r) - torch.log(f0)) / (2 * torch.log(sigma / f0))) ** 2)

                # Angular component (squared)
                g_angular = torch.exp(-((theta - theta0) ** 2) / (2 * sigma ** 2))

                # Combine radial and angular components
                g = g_radial * g_angular
                g = g * torch.cos(freq * r + psi)
                g = g / (2 * math.pi * (sigma ** 2))

                self.weight.data[i, j] = g

        self.conv.weight.data = self.weight.data

    def forward(self, x):
        if self.training or not self.is_calculated:
            self.calculate_weights()
            self.is_calculated = True

        if self.use_frequency_domain:
            # Convert to frequency domain
            x_freq = torch.fft.fft2(x)

            # Apply Log Gabor filter in frequency domain
            weight_freq = torch.fft.fft2(self.weight, s=x.shape[-2:])
            x_filtered_freq = x_freq * weight_freq

            # Additional frequency domain operations can be added here
            # For example, a simple low-pass filter:
            rows, cols = x.shape[-2:]
            crow, ccol = rows // 2, cols // 2
            mask = torch.zeros_like(x_filtered_freq)
            mask[:, :, crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
            x_filtered_freq = x_filtered_freq * mask

            # Convert back to spatial domain
            x_filtered = torch.fft.ifft2(x_filtered_freq).real

            # Apply any additional spatial domain operations
            x_filtered = F.conv2d(x_filtered, self.weight, stride=self.conv.stride,
                                  padding=self.conv.padding, dilation=self.conv.dilation,
                                  groups=self.conv.groups)
        else:
            # Spatial domain only
            x_filtered = self.conv(x)

        return x_filtered"""


"""
Overview of the EnhancedFrequencyLogGaborConv2d class:
    - Initialisation (init):
        - Sets up parameters for multi-scale Log Gabor filters.
        - Creates a frequency domain attention mechanism.
        - Initialises adaptive frequency-spatial mixing.
        - Sets up learnable frequency band selection.
        - Creates a spatial convolution layer. 
        
    - Log Gabor Filter Generation (get_log_gabor_filter):
        - Creates Log Gabor filters in the frequency domain for different scales.
    
    - Forward Pass (forward):
        - Converts input to frequency domain.
        - Applies multi-scale Log Gabor filtering.
        - Applies frequency domain attention.
        - Performs learnable frequency band selection.
        - Converts back to spatial domain.
        - Applies spatial convolution.
        - Mixes frequency and spatial domain results.
"""


class EnhancedFrequencyLogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", num_scales=3, device='cuda'):
        super(EnhancedFrequencyLogGaborConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_scales = num_scales
        self.device = device

        # Multi-scale Log Gabor parameters
        self.freq = nn.Parameter(torch.rand(num_scales, out_channels, in_channels, device=device) * math.pi)
        self.theta = nn.Parameter(torch.rand(num_scales, out_channels, in_channels, device=device) * math.pi)
        self.sigma = nn.Parameter(torch.rand(num_scales, out_channels, in_channels, device=device) * math.pi)
        self.f0 = nn.Parameter(torch.ones(num_scales, out_channels, in_channels, device=device))
        self.theta0 = nn.Parameter(torch.ones(num_scales, out_channels, in_channels, device=device))

        # Frequency domain attention
        self.freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels * num_scales, kernel_size=1),
            nn.BatchNorm2d(out_channels * num_scales),
            nn.Softmax(dim=1)
        ).to(device)

        # Adaptive frequency-spatial mixing
        self.mixing_param = nn.Parameter(torch.rand(1, device=device))

        # Learnable frequency band selection
        self.freq_band_select = nn.Parameter(torch.rand(out_channels, in_channels, 2, device=device))

        # Spatial convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode).to(device)

        # Register parameters
        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("f0", self.f0)
        self.register_parameter("theta0", self.theta0)
        self.register_parameter("mixing_param", self.mixing_param)
        self.register_parameter("freq_band_select", self.freq_band_select)

    """def get_log_gabor_filter(self, scale, size):
        y, x = torch.meshgrid([torch.linspace(-1, 1, size[0], device=self.device),
                               torch.linspace(-1, 1, size[1], device=self.device)],
                              indexing='ij')
        r = torch.sqrt(x**2 + y**2 + 1e-6)
        phi = torch.atan2(y, x)

        log_gabor = torch.exp(-((torch.log(r.unsqueeze(0).unsqueeze(0) - torch.log(self.f0[scale, :, :].unsqueeze(-1).unsqueeze(-1))))**2) /
                               (2 * (torch.log(self.sigma[scale, :, :].unsqueeze(-1).unsqueeze(-1)))**2))
        angular = torch.exp(-((phi.unsqueeze(0).unsqueeze(0) - self.theta[scale, :, :].unsqueeze(-1).unsqueeze(-1))**2) /
                            (2 * self.theta0[scale, :, :].unsqueeze(-1).unsqueeze(-1)**2))

        return log_gabor * angular"""

    def get_log_gabor_filter(self, scale, size):
        y, x = torch.meshgrid([torch.linspace(-1, 1, size[0], device=self.device),
                               torch.linspace(-1, 1, size[1], device=self.device)],
                              indexing='ij')
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-6)
        phi = torch.atan2(y, x)

        r = r.unsqueeze(0).unsqueeze(0)
        phi = phi.unsqueeze(0).unsqueeze(0)

        f0 = self.f0[scale, :, :].unsqueeze(-1).unsqueeze(-1)
        sigma = self.sigma[scale, :, :].unsqueeze(-1).unsqueeze(-1)
        theta = self.theta[scale, :, :].unsqueeze(-1).unsqueeze(-1)
        theta0 = self.theta0[scale, :, :].unsqueeze(-1).unsqueeze(-1)
        freq = self.freq[scale, :, :].unsqueeze(-1).unsqueeze(-1)

        log_gabor = torch.exp(-((torch.log(r) - torch.log(f0)) ** 2) / (2 * (torch.log(sigma / f0)) ** 2))
        # log_gabor = torch.exp(-((torch.log(r.to(self.device)) - torch.log(f0.to(self.device))) ** 2) / (
        #             2 * (torch.log(sigma.to(self.device) / f0.to(self.device))) ** 2))  # 16_07_2024

        angular = torch.exp(-((phi - theta) ** 2) / (2 * theta0 ** 2))

        g = log_gabor * angular
        g = g * torch.cos(freq * r)
        g = g / (2 * math.pi * sigma ** 2)

        return g

    def forward(self, x):
        x = x.to(self.device)
        batch_size, _, height, width = x.shape

        # Convert to frequency domain
        x_freq = torch.fft.fft2(x)
        # x_freq = torch.fft.fft2(x.to(self.device))  # 16_07_2024
        x_freq_shift = torch.fft.fftshift(x_freq)

        """# Multi-scale Log Gabor filtering in frequency domain
        filtered_outputs = []
        for scale in range(self.num_scales):
            scale_filter = self.get_log_gabor_filter(scale, (height, width))
            filtered = x_freq_shift.unsqueeze(1) * scale_filter.unsqueeze(0)
            filtered_outputs.append(filtered)"""

        # Compute filters once (debug)
        filters = [self.get_log_gabor_filter(scale, (height, width)) for scale in range(self.num_scales)]

        # Apply filters in frequency domain (Debug)
        filtered_outputs = [x_freq_shift.unsqueeze(1) * filter.unsqueeze(0) for filter in filters]

        # Frequency domain attention
        attention_weights = self.freq_attention(torch.abs(x_freq_shift))
        attention_weights = attention_weights.view(batch_size, self.num_scales, self.out_channels, height, width)

        # Apply attention and sum
        attended_output = torch.zeros_like(x_freq_shift, dtype=torch.complex64)

        for scale, (attention, filtered) in enumerate(zip(attention_weights.unbind(1), filtered_outputs)):
            # Reshape attention to match filtered
            attention = attention.unsqueeze(2).expand(-1, -1, 3, -1, -1)
            attended_output += (attention * filtered).sum(dim=1)

        # Reshape attended_output to match x_freq_shift
        attended_output = attended_output.sum(dim=1)  # Sum over out_channels

        # Learnable frequency band selection
        freq_mask = torch.zeros((self.out_channels, self.in_channels, height, width), device=self.device)
        # freq_mask = torch.zeros((self.out_channels, self.in_channels, height, width)).to(self.device)  # 16_07_2024
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                start, end = self.freq_band_select[i, j]
                start_idx = int((start + 1) / 2 * height)
                end_idx = int((end + 1) / 2 * height)
                freq_mask[i, j, start_idx:end_idx, start_idx:end_idx] = 1
        freq_mask = freq_mask.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        attended_output = attended_output.unsqueeze(1) * freq_mask.unsqueeze(2)
        # attended_output = torch.zeros_like(x_freq_shift, dtype=torch.complex64).to(self.device)  # 16_07_2024

        # Convert back to spatial domain
        x_filtered = torch.fft.ifft2(torch.fft.ifftshift(attended_output)).real

        # Adjust dimensions: sum over extra dimensions
        x_filtered = x_filtered.sum(dim=(2, 3))  # Sum over the extra dimensions

        # Reshape x_filtered to match x_spatial
        x_filtered = x_filtered.view(batch_size, self.out_channels, height, width)

        # Spatial domain convolution
        x_spatial = self.conv(x)

        # Adaptive frequency-spatial mixing
        output = self.mixing_param * x_filtered + (1 - self.mixing_param) * x_spatial

        return output

    def to(self, device):
        self.device = device
        # self.freq_attention = self.freq_attention.to(device)  # 16_07_2024
        # self.conv = self.conv.to(device)  # 16_07_2024
        return super(EnhancedFrequencyLogGaborConv2d, self).to(device)
