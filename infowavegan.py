import torch
import torch.nn.functional as F
import numpy as np

class UpConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=25,
        stride=4,
        padding='zeros',
        upsample='zeros',
        relu=True,
        use_batchnorm=False
    ):
        self.relu = relu
        super(UpConv, self).__init__()
        self.conv = torch.nn.ConvTranspose1d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=(kernel_size),
                        stride=stride,
                        padding=11,
                        output_padding=1,
                    )
        self.batch_norm = torch.nn.BatchNorm1d(filters) if use_batchnorm else torch.nn.Identity()

    def forward(self, x):
        output = self.conv(x)
        output = self.batch_norm(output)
        output = F.relu(output) if self.relu else torch.tanh(output)
        return(output)

class DownConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=25,
        stride=4,
        use_batchnorm=False,
        alpha=0.2,
        phaseshuffle_rad=0
    ):
        super(DownConv, self).__init__()
        self.alpha = alpha
        self.phaseshuffle_rad = phaseshuffle_rad

        self.conv = torch.nn.Conv1d(
                        in_channels,
                        filters,
                        kernel_size,
                        stride,
                        padding=11
                    )
        self.batchnorm = torch.nn.BatchNorm1d(filters) if use_batchnorm else torch.nn.Identity()

    def forward(self, x):
        output = self.conv(x)
        output = self.batchnorm(output)
        output = F.leaky_relu(output, self.alpha)
        output = self.phaseshuffle(output, self.phaseshuffle_rad)
        return output

    def phaseshuffle(self, x, rad):
        phase = np.random.randint(-rad, rad+1)
        pad_l = np.max(phase, 0)
        pad_r = np.max(-phase, 0)
        shuffle = torch.nn.ReflectionPad1d((pad_l, pad_r))
        x = shuffle(x)
        return(x)

class WaveGANGenerator(torch.nn.Module):
    def __init__(
        self,
        slice_len=16384,
        nch=1,
        kernel_len=25,
        stride=4,
        dim=64,
        use_batchnorm=False,
        latent_dim=100,
        upsample='zeros',
        train=False
    ):
        assert slice_len in [16384]
        super(WaveGANGenerator, self).__init__()
        dim_mul = 16 if slice_len == 16384 else 32
        self.dim = dim
        self.dim_mul = dim_mul

        # [100] -> [16, 1024]
        self.z_project = torch.nn.Linear(latent_dim, 4 * 4 * dim * dim_mul)
        self.z_batchnorm = torch.nn.BatchNorm1d(dim*dim_mul) if use_batchnorm else torch.nn.Identity()
        dim_mul //= 2

        # [16, 1024] -> [64, 512]
        self.upconv0 = UpConv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

        # [64, 512] -> [256, 256]
        self.upconv1 = UpConv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

        # [256, 256] -> [1024, 128]
        self.upconv2 = UpConv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

        # [1024, 128] -> [4096, 64]
        self.upconv3 = UpConv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )

        # [4096, 64] -> [16384, nch]
        self.upconv4 = UpConv(
                        dim * dim_mul,
                        nch,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm,
                        relu=False
                       )

    def forward(self, z):
        # Project and reshape
        output = self.z_project(z)
        output = self.z_batchnorm(output.view(-1, self.dim * self.dim_mul, 16))

        # Conv layers
        output = self.upconv0(output)
        output = self.upconv1(output)
        output = self.upconv2(output)
        output = self.upconv3(output)
        output = self.upconv4(output)
        return(output)

class WaveGANDiscriminator(torch.nn.Module):
    def __init__(
        self,
        kernel_len=25,
        dim=64,
        stride=4,
        use_batchnorm=False,
        phaseshuffle_rad=0
    ):
        super(WaveGANDiscriminator, self).__init__()
        self.dim=dim

        # Conv Layers
        self.downconv_0 = DownConv(1, dim, kernel_len, stride, use_batchnorm, phaseshuffle_rad)
        self.downconv_1 = DownConv(dim, dim*2, kernel_len, stride, use_batchnorm, phaseshuffle_rad)
        self.downconv_2 = DownConv(dim*2, dim*4, kernel_len, stride, use_batchnorm, phaseshuffle_rad)
        self.downconv_3 = DownConv(dim*4, dim*8, kernel_len, stride, use_batchnorm, phaseshuffle_rad)
        self.downconv_4 = DownConv(dim*8, dim*16, kernel_len, stride, use_batchnorm, phaseshuffle_rad)

        # Logit
        self.fc_out = torch.nn.Linear(dim*16*16, 1)

    def forward(self, x):
        output = self.downconv_0(x)
        output = self.downconv_1(output)
        output = self.downconv_2(output)
        output = self.downconv_3(output)
        output = self.downconv_4(output)
        output = self.fc_out(output.view(-1, self.dim*16*16))
        return output

class WaveGANQNetwork(WaveGANDiscriminator):
    def __init__(
        self,
        num_categ,
        kernel_len=25,
        dim=64,
        stride=4,
        use_batchnorm=False,
        phaseshuffle_rad=0,
    ):
        super(WaveGANQNetwork, self).__init__(
                                        kernel_len=25,
                                        dim=64,
                                        stride=4,
                                        use_batchnorm=False,
                                        phaseshuffle_rad=0
                                    )
        self.fc_out = torch.nn.Linear(dim*16*16, num_categ)

# z = torch.Tensor(np.random.uniform(-1, 1, (25, 100)))
# G = WaveGANGenerator()
# D = WaveGANDiscriminator(phaseshuffle_rad=20)
# Q = WaveGANQNetwork(latent_dim=10, phaseshuffle_rad=20)
# G_z = G(z)
# D_G_z = D(G_z)
# Q_G_z = Q(G_z)
