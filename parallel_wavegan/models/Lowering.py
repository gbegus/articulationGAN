"""
Harmonic Lowering
author: Hirotoshi Takeuchi
"""
import warnings

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import parallel_wavegan.models.interp_function as IF

__all__ = ["HarmonicLowering", "LogHarmonicLowering"]

class BaseLowering(nn.Module):
    def __init__(
        self,
        anchor,
        f_kernel_size,
        in_channels,
        groups=1,
        ):
        super().__init__()
        self.anchor = anchor
        self.f_kernel_size = f_kernel_size
        self.in_channels = in_channels
        self.groups = groups

        # setting index rules of in channels
        self.channel_slice_func = lambda k_f: slice(
            k_f*self.in_channels, (k_f+1)*self.in_channels)
        self.channel_type = "stack"
        if self.groups != 1:
            warnings.warn(
                "Harmonic Lowering implementation can be not good at group convolution")
            self.channel_slice_func = lambda k_f: slice(
                k_f, None, self.in_channels)
            self.channel_type = "textile"

        # make parallel streams
        self.parallel_streams = [torch.cuda.Stream() for _ in range(self.f_kernel_size)]

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        lowered_in_channels = in_channels * f_kernel_size
        return Tensor size(batch,lowered_in_channels,f,t)
        """
        # make empty lowered input
        batch, in_channels, in_freq_size, in_time_size = input.shape
        lowered_in_channels = in_channels * self.f_kernel_size
        lowered_shape = (batch, lowered_in_channels,
                         in_freq_size, in_time_size)
        lowered_input = torch.empty(
            lowered_shape, dtype=input.dtype, device=input.device)
        # fill elements start
        current_stream = torch.cuda.current_stream()
        #   block current stream
        current_stream.synchronize()
        for fk in range(self.f_kernel_size):
            # start parallel
            with torch.cuda.stream(self.parallel_streams[fk]):
                lowered_input[:,self.channel_slice_func(fk)] = self.parallelized(input,k=fk+1)

        for s in self.parallel_streams:
            # block parallel streams
            s.synchronize()

        # fill elements end
        return lowered_input

    def parallelized(self,input,k):
        raise NotImplementedError

    def extra_repr(self):
        return f"channel_type={self.channel_type}"


class HarmonicLowering(BaseLowering):
    """
    Lowering input for normal convolution
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.zoom = IF.Zoom()

    def parallelized(self,input,k):
        return self.zoom(input,k,self.anchor)

    def extra_repr(self):
        return f"n={self.anchor}, K_f={self.f_kernel_size}, " + super().extra_repr()

class LogHarmonicLowering(BaseLowering):
    def __init__(
        self,
        anchor,
        f_kernel_size,
        in_channels,
        groups=1,
        out_log_scale=1000,
        in_log_scale=0.001,
        radix=None,
        ):
        super().__init__(anchor,f_kernel_size,in_channels,groups)
        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix
        self.shift = self.make_log_shift()
        self.Shifter = IF.Shift()

    def make_log_shift(self):
        """
        compute log shift
        return ndarray size(f_kernel_size)
        """
        assert 1 <= self.anchor <= self.f_kernel_size, f"invalid anchor={self.anchor}. anchor should be in [min=1,f_kernel_size={self.f_kernel_size}]"

        np_shift = (np.arange(self.f_kernel_size) + 1)/self.anchor
        if self.radix is None:
            log_shift = self.out_log_scale * np.log(self.in_log_scale * np_shift)
        else:
            log_shift = self.out_log_scale * \
                np.log(self.in_log_scale * np_shift) / np.log(self.radix)
        target_index = self.anchor - 1
        log_shift -= log_shift[target_index]
        return -log_shift

    def parallelized(self,input,k):
        return self.Shifter(input,self.shift[k-1])

    def extra_repr(self):
        radix = self.radix if self.radix is not None else "e"
        return f"n={self.anchor}, K_f={self.f_kernel_size}, log_func(f)={self.out_log_scale}log_{radix} {self.in_log_scale}f, " + super().extra_repr() 
