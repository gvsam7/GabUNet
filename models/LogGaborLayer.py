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
                # Log-Gabor filter
                # g = torch.exp(-1 * ((torch.log(r) - torch.log(f0)) / (2 * torch.log(sigma / f0))) ** 2)
                # Bi-dimensional Log-Gabor filter
                # g = torch.exp(-1 * ((torch.log(r) - torch.log(f0)) / (2 * torch.log(sigma / f0))) ** 2)
                # g = g * torch.exp(-(theta - theta0) / (2 * sigma ** 2))  # Adjusted term
                # Log-Gabor filter (radial component)
                g_radial = torch.exp(-1 * ((torch.log(r) - torch.log(f0)) / (2 * torch.log(sigma / f0))) ** 2)

                # Angular component (squared as suggested)
                g_angular = torch.exp(-((theta - theta0) ** 2) / (2 * sigma ** 2))

                # Combine radial and angular components
                g = g_radial * g_angular

                g = g * torch.cos(freq * r + psi)
                g = g / (2 * math.pi * (sigma ** 2))

                self.conv.weight.data[i, j] = g


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

        # Initialize grid parameters
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

        return x_filtered


class EnhancedFrequencyLogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", num_scales=3):
        super(EnhancedFrequencyLogGaborConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_scales = num_scales

        # Multi-scale Log Gabor parameters
        self.freq = nn.Parameter(torch.rand(num_scales, out_channels, in_channels) * math.pi)
        self.theta = nn.Parameter(torch.rand(num_scales, out_channels, in_channels) * math.pi)
        self.sigma = nn.Parameter(torch.rand(num_scales, out_channels, in_channels) * math.pi)
        self.f0 = nn.Parameter(torch.ones(num_scales, out_channels, in_channels))
        self.theta0 = nn.Parameter(torch.ones(num_scales, out_channels, in_channels))

        # Frequency domain attention
        self.freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Adaptive frequency-spatial mixing
        self.mixing_param = nn.Parameter(torch.rand(1))

        # Learnable frequency band selection
        self.freq_band_select = nn.Parameter(torch.rand(out_channels, in_channels, 2))  # Start and end of band

        # Spatial convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def get_log_gabor_filter(self, scale, i, j, size):
        y, x = torch.meshgrid([torch.linspace(-1, 1, size[0]), torch.linspace(-1, 1, size[1])])
        r = torch.sqrt(x**2 + y**2 + self.freq.data.new_tensor(1e-6))
        phi = torch.atan2(y, x)

        log_gabor = torch.exp(-((torch.log(r) - torch.log(self.f0[scale, i, j]))**2) / (2 * (torch.log(self.sigma[scale, i, j]))**2))
        angular = torch.exp(-((phi - self.theta[scale, i, j])**2) / (2 * self.theta0[scale, i, j]**2))

        return log_gabor * angular

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Convert to frequency domain
        x_freq = torch.fft.fft2(x)
        x_freq_shift = torch.fft.fftshift(x_freq)

        # Multi-scale Log Gabor filtering in frequency domain
        filtered_outputs = []
        for scale in range(self.num_scales):
            scale_filter = torch.stack([self.get_log_gabor_filter(scale, i, j, (height, width))
                                        for i in range(self.out_channels)
                                        for j in range(self.in_channels)])
            scale_filter = scale_filter.view(self.out_channels, self.in_channels, height, width)
            filtered = x_freq_shift * scale_filter.unsqueeze(0)
            filtered_outputs.append(filtered)

        # Frequency domain attention
        attention_weights = self.freq_attention(torch.abs(x_freq_shift))
        attended_output = sum([w * f for w, f in zip(attention_weights.unbind(1), filtered_outputs)])

        # Learnable frequency band selection
        freq_mask = torch.zeros_like(attended_output)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                start, end = self.freq_band_select[i, j]
                start_idx = int((start + 1) / 2 * height)
                end_idx = int((end + 1) / 2 * height)
                freq_mask[i, j, start_idx:end_idx, start_idx:end_idx] = 1
        attended_output = attended_output * freq_mask

        # Convert back to spatial domain
        x_filtered = torch.fft.ifft2(torch.fft.ifftshift(attended_output)).real

        # Spatial domain convolution
        x_spatial = self.conv(x)

        # Adaptive frequency-spatial mixing
        output = self.mixing_param * x_filtered + (1 - self.mixing_param) * x_spatial

        return output
