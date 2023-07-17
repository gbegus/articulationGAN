# -*- coding: utf-8 -*-

"""A2M Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.layers import WNConv1d, GBlock
from parallel_wavegan.utils import read_hdf5
from parallel_wavegan.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)


class A2MGenerator2(torch.nn.Module):
    """A2M generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        all_channels=None,
        kernel_size=7,
        out_kernel_size=7,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        middle_kernel_sizes=[],
        do_transpose=False,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        encoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_macaron_style_in_conformer: bool = True,
        conformer_rel_pos_type: str = "latest", # "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        extra_art: bool = False,
    ):
        """Initialize A2MGenerator2 module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            use_macaron_style_in_conformer: Whether to use macaron style FFN.
            conformer_rel_pos_type (str): Relative pos encoding type in conformer.
            conformer_pos_enc_layer_type (str): Pos encoding layer type in conformer.
            conformer_self_attn_layer_type (str): Self-attention layer type in conformer
            conformer_activation_type (str): Activation function type in conformer.
            use_cnn_in_conformer: Whether to use CNN in conformer.
            zero_triu: Whether to use zero triu in relative self-attention module.
            conformer_enc_kernel_size: Kernel size of encoder conformer.

        """
        super().__init__()

        if extra_art:
            self.input_conv = WNConv1d(in_channels, channels, kernel_size=2)
        else:
            assert kernel_size % 2 == 1, "Kernel size must be odd number."
            self.input_conv = WNConv1d(
                in_channels,
                channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )
        
        tmp = [GBlock(channels, channels, kernel_size=kernel_size), \
            GBlock(channels, adim, kernel_size=kernel_size)]
        self.middle_convs = torch.nn.ModuleList(tmp)
        
        if conformer_rel_pos_type == "legacy":
            if conformer_pos_enc_layer_type == "rel_pos":
                conformer_pos_enc_layer_type = "legacy_rel_pos"
                logging.warning(
                    "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                    "due to the compatibility. If you want to use the new one, "
                    "please use conformer_pos_enc_layer_type = 'latest'."
                )
            if conformer_self_attn_layer_type == "rel_selfattn":
                conformer_self_attn_layer_type = "legacy_rel_selfattn"
                logging.warning(
                    "Fallback to "
                    "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                    "due to the compatibility. If you want to use the new one, "
                    "please use conformer_pos_enc_layer_type = 'latest'."
                )
        elif conformer_rel_pos_type == "latest":
            assert conformer_pos_enc_layer_type != "legacy_rel_pos"
            assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
        else:
            raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")
        
        encoder_input_layer = torch.nn.Identity()
        self.encoder = ConformerEncoder(
            idim=adim, # Dimension of the inputs.
            attention_dim=adim, # 384
            attention_heads=aheads, # 4
            linear_units=eunits, # 1536, Number of encoder hidden units.
            num_blocks=elayers, # 6, Number of encoder layers.
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            pos_enc_layer_type=conformer_pos_enc_layer_type,
            selfattention_layer_type=conformer_self_attn_layer_type,
            activation_type=conformer_activation_type,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_enc_kernel_size,
            zero_triu=zero_triu,
        )

        self.output_fc = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adim, out_channels),
            torch.nn.Tanh(),
        )

        # apply weight norm
        # if use_weight_norm:
        #     self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, ar=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Return:
            Tensor: Output tensor (B, out_channels, T).
        """
        c = self.input_conv(c)
        for l_conv in self.middle_convs:
            c = l_conv(c)
        xs = c.permute(0,2,1)  # (batch_size, time, idim)
        x_masks = torch.ones((xs.shape[0], 1, xs.shape[1]), dtype=torch.uint8).to(xs.device)
        hs, _ = self.encoder(xs, x_masks)  # (batch_size, time, adim)
        c = self.output_fc(hs)  # (batch_size, time, num_feats)
        c = c.permute(0,2,1)  # (batch_size, num_feats, time=input_len*scale), e.g. scale = 110
        return c

    def reset_parameters(self):
        """Reset parameters.

        Using HiFi-GAN's initialization:
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

    def inference(self, c, normalize_before=False):
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
