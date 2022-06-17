# -*- coding: utf-8 -*-

"""HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from articulatory.layers import HiFiGANResidualBlock as ResidualBlock
from articulatory.layers import WNConv1d, PastFCEncoder, GBlock
from articulatory.utils import read_hdf5


class HiFiMultGenerator(torch.nn.Module):
    """HiFiGAN generator module."""

    def __init__(
        self,
        in_list=[80],
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        final_scale=110,
        upsample_kernel_sizes=(16, 16, 4, 4),
        paddings=None,
        output_paddings=None,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_ar=False, ar_input=512, ar_hidden=256, ar_output=128,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
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

        self.use_ar = use_ar

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
        
        if paddings is None:
            paddings = [upsample_scales[i] // 2 + upsample_scales[i] % 2 for i in range(len(upsample_kernel_sizes))]
        else:
            new_paddings = []
            for i, s in enumerate(paddings):
                if s == "default":
                    new_paddings.append(upsample_scales[i] // 2 + upsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            paddings = new_paddings
        if output_paddings is None:
            output_paddings = [upsample_scales[i] % 2 for i in range(len(upsample_kernel_sizes))]
        else:
            new_output_paddings = []
            for i, s in enumerate(output_paddings):
                if s == "default":
                    new_output_paddings.append(upsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            output_paddings = new_output_paddings
        self.final_scale = final_scale

        self.encoders = torch.nn.ModuleList()
        for in_channels in in_list:
            self.encoders += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels,
                        channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    GBlock(channels, channels, kernel_size=kernel_size, norm=False),
                )
            ]

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)

        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            # assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=paddings[i],
                        output_padding=output_paddings[i],
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        self.ar_model = PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, xs, spk_id=None, ar=None):
        """Calculate forward propagation.

        Args:
            xs: List of Tensor objects
                Tensor i has shape (B_i, in_channels_i, T_in).
                can be None instead of a Tensor
            ar: Tensor, ar.shape[0] = sum([B_i])

        Returns:
            outs: Tensor with shape (sum([B_i]), out_channels, T_out).

        """
        outs = []
        for mod_i, x in enumerate(xs):
            if x is not None:
                ar_si = 0
                if self.use_ar:
                    ar_feats = self.ar_model(ar[ar_si:ar_si+len(x)])  # (batchsize, ar_output)
                    ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, x.shape[2])  # (batchsize, ar_output, length)
                    x = torch.cat((x, ar_feats), dim=1)
                    ar_si += len(x)
                c = self.encoders[mod_i](x)
                
                for i in range(self.num_upsamples):
                    c = self.upsamples[i](c)
                    cs = 0.0  # initialize
                    for j in range(self.num_blocks):
                        cs += self.blocks[i * self.num_blocks + j](c)
                    c = cs / self.num_blocks
                out = self.output_conv(c)  # (batch_size, 1, input_len*pre_final_scale)
                outs.append(out)
        outs = torch.cat(outs, dim=0)
        return outs

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

    def inference(self, c, normalize_before=False):  # NOTE unimplemented
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)
