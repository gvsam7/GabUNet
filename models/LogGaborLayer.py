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
        self.f0 = Parameter(torch.ones(out_channels, in_channels), requires_grad=True)
        self.theta0 = Parameter(torch.ones(out_channels, in_channels), requires_grad=True)


        # self.f0 = Parameter(torch.Tensor([1.0]), requires_grad=True)  # Define f0 as a parameter

        # self.theta0 = Parameter(torch.Tensor([1.0]), requires_grad=True)  # Define theta0 as a parameter

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
                # f0 = self.f0.expand_as(self.y)  # Expand f0 accordingly
                # theta0 = self.theta0.expand_as(self.y)
                f0 = self.f0[i, j].expand_as(self.y)
                theta0 = self.theta0[i, j].expand_as(self.y)

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
