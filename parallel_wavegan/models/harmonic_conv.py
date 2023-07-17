"""
Harmonic Convolution implementation by Harmonic Lowering in pytorch
author: Hirotoshi Takeuchi
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel_wavegan.models.Lowering import HarmonicLowering
from parallel_wavegan.models.Lowering import LogHarmonicLowering

__all__ = ["SingleHarmonicConv2d", "SingleLogHarmonicConv2d"]


class BaseSingleHarmonicConv2d_ft(nn.Conv2d):
    """
    Base class for Harmonic Convolution
    """
    
    def __init__(self, *args, anchor=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(anchor, int):
            raise Exception("anchor should be integer")
        self.anchor = anchor

        if self.anchor < 1:
            raise Exception("anchor should be equal to or bigger than 1")
        if self.padding_mode != "zeros":
            raise NotImplementedError("only zero padding mode is implemented")

        if self.padding[0] != 0:
            warnings.warn(
                "Harmonic Convolution do no padding on frequency axis")
            self.padding = (0,self.padding[1])

        # transforming weight shape
        lowered_shape = (self.out_channels,self.in_channels*self.kernel_size[0],1,self.kernel_size[1])
        self.lowered_weight = torch.nn.Parameter(self.weight.reshape(lowered_shape))
        self.weight = None

    def forward(self, input):
        # [batch, in_channel, f, t]
        raise NotImplementedError("overwrite forward method")


class SingleHarmonicConv2d(BaseSingleHarmonicConv2d_ft):
    """
    Harmonic Convolution by Harmonic Lowering
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.HL = HarmonicLowering(
            anchor=self.anchor,
            f_kernel_size=self.kernel_size[0],
            in_channels=self.in_channels,
            groups=self.groups,
        )

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        return Tensor size(batch,out_channels,f',t')
        """
        lowered_input = self.HL(input)

        output = F.conv2d(
            input=lowered_input,
            weight=self.lowered_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return f"anchor={self.anchor}, " + super().extra_repr()


class SingleLogHarmonicConv2d(BaseSingleHarmonicConv2d_ft):
    def __init__(self, *args, out_log_scale=1000, in_log_scale=0.001, radix=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix

        self.LHL = LogHarmonicLowering(
            anchor=self.anchor,
            f_kernel_size=self.kernel_size[0],
            in_channels=self.in_channels,
            groups=self.groups,
            out_log_scale=self.out_log_scale,
            in_log_scale=self.in_log_scale,
            radix=self.radix,
        )

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        return Tensor size(batch,out_channels,f',t')
        """
        lowered_input = self.LHL(input)

        output = F.conv2d(
            input=lowered_input,
            weight=self.lowered_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return f"out_log_scale={self.out_log_scale}, in_log_scale={self.in_log_scale}, radix={self.radix}, " + super().extra_repr()
