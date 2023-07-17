# -*- coding: utf-8 -*-

"""HiFi-GAN-based articulatory inversion model.

Copyright 2022 Peter Wu
MIT License (https://opensource.org/licenses/MIT)
"""

import logging

import numpy as np
import torch

from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.layers import PastFCEncoder
from parallel_wavegan.utils import read_hdf5


class HiFiInvGenerator(torch.nn.Module):
    """HiFi-GAN-Inverse generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        downsample_scales=(8, 8, 2, 2),
        downsample_kernel_sizes=(16, 16, 4, 4),
        paddings=None,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        extra_art=False,
        use_ar=False, ar_input=512, ar_hidden=256, ar_output=128,
        use_tanh=True
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            downsample_scales (list): List of upsampling scales.
            downsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.extra_art = extra_art
        self.use_ar = use_ar

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(downsample_scales) == len(downsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
        
        if paddings is None:
            paddings = [downsample_scales[i] // 2 + downsample_scales[i] % 2 for i in range(len(downsample_kernel_sizes))]
        else:
            new_paddings = []
            for i, s in enumerate(paddings):
                if s == "default":
                    new_paddings.append(downsample_scales[i] // 2 + downsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            paddings = new_paddings

        # define modules
        self.num_downsamples = len(downsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.downsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(downsample_kernel_sizes)):
            # assert downsample_kernel_sizes[i] == 2 * downsample_scales[i]
            self.downsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.Conv1d(
                        channels * (2 ** i),
                        channels * (2 ** (i + 1)),
                        downsample_kernel_sizes[i],
                        stride=downsample_scales[i],
                        padding=paddings[i],
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels * (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        if not extra_art:
            if use_tanh:
                self.output_conv = torch.nn.Sequential(
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        channels * (2 ** (i + 1)),
                        out_channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.Tanh(),
                )
            else:
                self.output_conv = torch.nn.Sequential(
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        channels * (2 ** (i + 1)),
                        out_channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                    ),
                )
        else:
            if use_tanh:
                self.output_conv = torch.nn.Sequential([
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(in_channels, in_channels, kernel_size=2, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        channels * (2 ** (i + 1)),
                        out_channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.Tanh(),
                ])
            else:
                self.output_conv = torch.nn.Sequential([
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(in_channels, in_channels, kernel_size=2, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        channels * (2 ** (i + 1)),
                        out_channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                    ),
                ])

        self.ar_model = PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, ar=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        if self.use_ar:
            ar_feats = self.ar_model(ar) # (batchsize, ar_output)
            ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, c.shape[2]) # (batchsize, ar_output, length)
            c = torch.cat((c, ar_feats), dim=1)
        c = self.input_conv(c)
        # print('after input_conv', c.shape)
        for i in range(self.num_downsamples):
            c = self.downsamples[i](c)
            # print('after upsample %d' % i, c.shape)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            # print('cs', cs.shape)
            c = cs / self.num_blocks
        c = self.output_conv(c) # (batch_size, 1, input_len*final_scale)

        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(downsample_scales), out_channels).

        """
        c = c.unsqueeze(1)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")
