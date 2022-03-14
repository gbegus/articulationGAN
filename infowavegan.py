import torch
import torch.nn.functional as F
import numpy as np

class Upconv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=25,
        stride=4,
        padding='zeros',
        upsample='zeros',
        use_batchnorm=False
    ):
        super(Upconv, self).__init__()
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
        output = F.relu(output)
        return(output)

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


        #  Reshape
        # [100] -> [16, 1024]
        self.z_project = torch.nn.Linear(latent_dim, 4 * 4 * dim * dim_mul)
        self.z_batchnorm = torch.nn.BatchNorm1d(dim*dim_mul) if use_batchnorm else torch.nn.Identity()
        dim_mul //= 2

        # Layer 0
        # [16, 1024] -> [64, 512]
        self.upconv0 = Upconv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

        # Layer 1
        # [64, 512] -> [256, 256]
        self.upconv1 = Upconv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

        # Layer 2
        # [256, 256] -> [1024, 128]
        self.upconv2 = Upconv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

        # Layer 3
        # [1024, 128] -> [4096, 64]
        self.upconv3 = Upconv(
                        dim * dim_mul * 2,
                        dim * dim_mul,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )

        # Layer 4
        # [4096, 64] -> [16384, nch]
        self.upconv4 = Upconv(
                        dim * dim_mul,
                        nch,
                        kernel_len,
                        stride,
                        use_batchnorm=use_batchnorm
                       )
        dim_mul //= 2

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

# z = torch.Tensor(np.random.uniform(-1, 1, (25, 100)))
# G = WaveGANGenerator()
# print(G(z).shape)
