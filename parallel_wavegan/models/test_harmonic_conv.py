import torch
import harmonic_conv


a = torch.ones(16, 1, 256, 100) # batch_size, 1, num_feats, seq_len
    # num_feats is treated as the frequency dim
a = a.cuda()
conv_module = harmonic_conv.SingleHarmonicConv2d(1, 5, (5, 3), anchor=4, stride=1, padding=(0, 1))
    # for kernel (k1, k2), k1 is for num_feats and k2 is for seq_len
conv_module = conv_module.cuda()
with torch.no_grad():
    b = conv_module(a)
    print(b.shape) # 16, 5, 256, 98
# outputed num_feats dim always seems to be the same as the inputed one
#   holds for anchor={1,2,4} and prob all the others eg 3 and 5, 6, ...

# TODO decide if i should do weighted sum of anchors 1 through 7
# weighted sum mixing is implemented as an extra 1×1 convolution
