"""
Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf 
Original code:
    https://github.com/dgaddy/subvocal/blob/master/transduction_model.py
"""

import logging
import numpy as np
import random
import torch
import torch.nn.functional as F

from torch import nn

from parallel_wavegan.layers import WNConv1d, ResBlock, TransformerEncoderLayer
from parallel_wavegan.utils import read_hdf5


class A2MTransformer(nn.Module):
    def __init__(self, in_channels=8, out_channels=80, elayers=6, hidden_dim=768, dropout=.2, extra_art=False,
    use_ar=False, ar_input=512, ar_hidden=256, ar_output=128, 
    use_tanh=False):
        super().__init__()

        if extra_art:
            self.conv_blocks = nn.Sequential(
                WNConv1d(in_channels, hidden_dim, kernel_size=2),
                ResBlock(hidden_dim, hidden_dim, 1),
                ResBlock(hidden_dim, hidden_dim, 1),
                ResBlock(hidden_dim, hidden_dim, 1),
            )
        else:
            self.conv_blocks = nn.Sequential(
                ResBlock(in_channels, hidden_dim, 1),
                ResBlock(hidden_dim, hidden_dim, 1),
                ResBlock(hidden_dim, hidden_dim, 1),
            )
        self.w_raw_in = nn.Linear(hidden_dim, hidden_dim)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, elayers)
        self.w_out = nn.Linear(hidden_dim, out_channels)
        # self.w_aux = nn.Linear(hidden_dim, num_aux_outs)

    def forward(self, x, ar=None):
        '''
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        '''
        x = self.conv_blocks(x)
        x = x.transpose(1, 2)  # (batchsize, seq_len, num_feats)
        x = self.w_raw_in(x)
        x = x.transpose(0, 1)  # (seq_len, batchsize, num_feats)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batchsize, seq_len, num_feats)
        out = self.w_out(x)
        out = out.transpose(1, 2)
        return out

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
    
    def remove_weight_norm(self):
        pass

    def inference(self, x, normalize_before=False):
        x = x.unsqueeze(0) # 1, 246, 30
        x = x.permute(0, 2, 1) # 1, 30, 246
        return self.forward(x).squeeze(0).transpose(1, 0)


if __name__ == "__main__":
    t = torch.ones(16, 30, 100)
    model = A2MTransformer(in_channels=30, out_channels=80, elayers=6, hidden_dim=768, dropout=0.2, extra_art=True)
    out = model(t)
    print(out.shape)  # (16, 80, 99)
