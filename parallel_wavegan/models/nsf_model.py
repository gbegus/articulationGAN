import functools
import logging
import sys
import numpy as np
import time
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import yaml

from parallel_wavegan.layers import WNConv1d, GBlock, ResBlock, TransformerEncoderLayer, \
                                    PastFCEncoder, PastSeqEncoder
from parallel_wavegan.utils import read_hdf5


##########################################
# From NSF


##############
# Building blocks (torch.torch_nn modules + dimension operation)
        
#        
# 1D dilated convolution that keep the input/output length
class Conv1dKeepLength(torch_nn.Conv1d):
    """Causal convolution
    Input tensor:  (batchsize, dim_in, length)
    Output tensor: (batchsize, dim_out, length)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is optional
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant'):
        super(Conv1dKeepLength, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)

        self.pad_mode = pad_mode

        self.causal = causal
        # input & output length will be the same        
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le

        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()
        
    def forward(self, data):
        '''
        Args:
            data: (batchsize, dim_in, length)
        
        Return:
            output: (batchsize, dim_out, length)
        '''
        # add one dimension (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length)
        x = torch_nn_func.pad(data.unsqueeze(2), \
                              (self.pad_le, self.pad_ri, 0, 0),
                              mode = self.pad_mode).squeeze(2)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output

# 
# Moving average
class MovingAverage(Conv1dKeepLength):
    """ Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """
    def __init__(self, feature_dim, window_len, causal=False, \
                 pad_mode='replicate'):
        super(MovingAverage, self).__init__(
            feature_dim, feature_dim, 1, window_len, causal,
            groups=feature_dim, bias=False, tanh=False, \
            pad_mode=pad_mode)
        # set the weighting coefficients
        torch_nn.init.constant_(self.weight, 1/window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, data):
        '''
        Args:
            data: (batchsize, dim, length)
        
        Return:
            output: (batchsize, dim, length)
        '''
        return super(MovingAverage, self).forward(data)

# 
# FIR filter layer
class TimeInvFIRFilter(Conv1dKeepLength):
    """ Wrapper to define a FIR filter over Conv1d
        Note: FIR Filtering is conducted on each dimension (channel)
        independently: groups=channel_num in conv1d
    """                                                                  
    def __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False):
        """ __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False)
        feature_dim: dimension of input data
        filter_coef: 1-D tensor of filter coefficients
        causal: FIR is causal or not (default: true)
        flag_train: whether train the filter coefficients (default false)
        Input data: (batchsize=1, length, feature_dim)
        Output data: (batchsize=1, length, feature_dim)
        """
        super(TimeInvFIRFilter, self).__init__(
            feature_dim, feature_dim, 1, filter_coef.shape[0], causal,
            groups=feature_dim, bias=False, tanh=False)
        
        if filter_coef.ndim == 1:
            # initialize weight using provided filter_coef
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, 
                                        filter_coef.shape[0]])
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.torch_nn.Parameter(tmp_coef, 
                                                 requires_grad=flag_train)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)

    def forward(self, data):
        return super(TimeInvFIRFilter, self).forward(data)    

class TimeVarFIRFilter(torch_nn.Module):
    """ TimeVarFIRFilter
    Given sequences of filter coefficients and a signal, do filtering
    
    Filter coefs: (batchsize=1, signal_length, filter_order = K)
    Signal:       (batchsize=1, signal_length, 1)
    
    For batch 0:
     For n in [1, sequence_length):
       output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
       
    Note: filter coef (0, n, :) is only used to compute the output 
          at (0, n, 1)
    """
    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()
    
    def forward(self, signal, f_coef):
        """ 
        Filter coefs: (batchsize=1, signal_length, filter_order = K)
        Signal:       (batchsize=1, signal_length, 1)
        
        Output:       (batchsize=1, signal_length, 1)
        
        For n in [1, sequence_length):
          output(0, n, 1)= \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
          
        This method may be not efficient:
        
        Suppose signal [x_1, ..., x_N], filter [a_1, ..., a_K]
        output         [y_1, y_2, y_3, ..., y_N, *, * ... *]
               = a_1 * [x_1, x_2, x_3, ..., x_N,   0, ...,   0]
               + a_2 * [  0, x_1, x_2, x_3, ..., x_N,   0, ...,  0]
               + a_3 * [  0,   0, x_1, x_2, x_3, ..., x_N, 0, ...,  0]
        """
        signal_l = signal.shape[1]
        order_k = f_coef.shape[-1]

        # pad to (batchsize=1, signal_length + filter_order-1, dim)
        padded_signal = torch_nn_func.pad(signal, (0, 0, 0, order_k - 1))
        
        y = torch.zeros_like(signal)
        # roll and weighted sum, only take [0:signal_length]
        for k in range(order_k):
            y += torch.roll(padded_signal, k, dims=1)[:, 0:signal_l, :] \
                      * f_coef[:, :, k:k+1] 
        # done
        return y


# Sinc filter generator
class SincFilter(torch_nn.Module):
    """ SincFilter
        Given the cut-off-frequency, produce the low-pass and high-pass
        windowed-sinc-filters.
        If input cut-off-frequency is (batchsize=1, signal_length, 1),
        output filter coef is (batchsize=1, signal_length, filter_order).
        For each time step in [1, signal_length), we calculate one
        filter for low-pass sinc filter and another for high-pass filter.
        
        Example:
        import scipy
        import scipy.signal
        import numpy as np
        
        filter_order = 31
        cut_f = 0.2
        sinc_layer = SincFilter(filter_order)
        lp_coef, hp_coef = sinc_layer(torch.ones(1, 10, 1) * cut_f)
        
        w, h1 = scipy.signal.freqz(lp_coef[0, 0, :].numpy(), [1])
        w, h2 = scipy.signal.freqz(hp_coef[0, 0, :].numpy(), [1])
        plt.plot(w, 20*np.log10(np.abs(h1)))
        plt.plot(w, 20*np.log10(np.abs(h2)))
        plt.plot([cut_f * np.pi, cut_f * np.pi], [-100, 0])
    """
    def __init__(self, filter_order):
        super(SincFilter, self).__init__()
        # Make the filter oder an odd number
        #  [-(M-1)/2, ... 0, (M-1)/2]
        # 
        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 +1
        
    def hamming_w(self, n_index):
        """ prepare hamming window for each time step
        n_index (batchsize=1, signal_length, filter_order)
            For each time step, n_index will be [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 0, :] = [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 1, :] = [-(M-1)/2, ... 0, (M-1)/2]
            ...
        output  (batchsize=1, signal_length, filter_order)
            output[0, 0, :] = hamming_window
            output[0, 1, :] = hamming_window
            ...
        """
        # Hamming window
        return 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)
    
    def sinc(self, x):
        """ Normalized sinc-filter sin( pi * x) / pi * x
        https://en.wikipedia.org/wiki/Sinc_function
        
        Assume x (batchsize, signal_length, filter_order) and 
        x[0, 0, :] = [-half_order, - half_order+1, ... 0, ..., half_order]
        x[:, :, self.half_order] -> time index = 0, sinc(0)=1
        """
        y = torch.zeros_like(x)
        y[:,:,0:self.half_k]=torch.sin(np.pi * x[:, :, 0:self.half_k]) \
                              / (np.pi * x[:, :, 0:self.half_k])
        y[:,:,self.half_k+1:]=torch.sin(np.pi * x[:, :, self.half_k+1:]) \
                               / (np.pi * x[:, :, self.half_k+1:])
        y[:,:,self.half_k] = 1
        return y
        
    def forward(self, cut_f):
        """
        Args:
            cut_f: cut-off frequency, (batchsize, length, 1)
    
        Return:
            lp_coef: low-pass filter coefs  (batchsize, length, filter_order)
            hp_coef: high-pass filter coefs (batchsize, length, filter_order)
        """
        # create the filter order index
        with torch.no_grad():   
            # [- (M-1) / 2, ..., 0, ..., (M-1)/2]
            lp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            # [[[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ],
            #  [[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ]]
            lp_coef = lp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1) # batchsize, length, k
            
            hp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            hp_coef = hp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1) # batchsize, length, k
            
            # temporary buffer of [-1^n] for gain norm in hp_coef
            tmp_one = torch.pow(-1, hp_coef)
            
        # unnormalized filter coefs with hamming window
        lp_coef = cut_f * self.sinc(cut_f * lp_coef) \
                  * self.hamming_w(lp_coef)
        
        hp_coef = (self.sinc(hp_coef) \
                   - cut_f * self.sinc(cut_f * hp_coef)) \
                  * self.hamming_w(hp_coef)
        
        # normalize the coef to make gain at 0/pi is 0 dB
        # sum_n lp_coef[n]
        lp_coef_norm = torch.sum(lp_coef, axis=2).unsqueeze(-1)
        # sum_n hp_coef[n] * -1^n
        hp_coef_norm = torch.sum(hp_coef * tmp_one, axis=2).unsqueeze(-1)
        
        lp_coef = lp_coef / lp_coef_norm
        hp_coef = hp_coef / hp_coef_norm
        
        # return normed coef
        return lp_coef, hp_coef


# 
# Up sampling
class UpSampleLayer(torch_nn.Module):
    """ Wrapper over up-sampling
    Input tensor: (batchsize, dim, length)
    Ouput tensor: (batchsize, dim, length * up-sampling_factor)
    """
    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        # wrap a up_sampling layer
        self.scale_factor = up_sampling_factor
        self.l_upsamp = torch_nn.Upsample(scale_factor=self.scale_factor)
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = torch_nn.Identity()
            self.l_ave2 = torch_nn.Identity()
        return
    
    def forward(self, x): # x has shape (batchsize=1, dim, length)
        up_sampled_data = self.l_upsamp(x)

        # permute it backt to (batchsize=1, length, dim)
        # and do two moving average
        return self.l_ave1(self.l_ave2(up_sampled_data))
    

# Neural filter block (1 block)
class NeuralFilterBlock(torch_nn.Module):
    """ Wrapper over a single filter block
    """
    def __init__(self, signal_size, hidden_size,\
                 kernel_size=3, conv_num=10, layer_type='default'):
        super(NeuralFilterBlock, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.dilation_s = [np.power(2, x) for x in np.arange(conv_num)]

        # ff layer to expand dimension
        self.l_ff_1 = torch_nn.Linear(signal_size, hidden_size, \
                                      bias=False)
        self.l_ff_1_tanh = torch_nn.Tanh()
        
        # dilated conv layers
        if layer_type == 'default' or layer_type == 'gblock' or layer_type == 'transformer':
            tmp = [Conv1dKeepLength(hidden_size, hidden_size, x, \
                                    kernel_size, causal=True, bias=False) \
                for x in self.dilation_s]
        elif layer_type == 'gblock2':
            tmp = [GBlock(hidden_size, hidden_size, kernel_size=kernel_size)
                for x in self.dilation_s]
        else:
            logging.info("layer_type not supported")
            sys.exit(1)
        self.l_convs = torch_nn.ModuleList(tmp)
                
        # ff layer to de-expand dimension
        self.l_ff_2 = torch_nn.Linear(hidden_size, hidden_size//4, \
                                      bias=False)
        self.l_ff_2_tanh = torch_nn.Tanh()
        self.l_ff_3 = torch_nn.Linear(hidden_size//4, signal_size, \
                                      bias=False)
        self.l_ff_3_tanh = torch_nn.Tanh()        

        # a simple scale
        self.scale = torch_nn.Parameter(torch.tensor([0.1]), 
                                        requires_grad=False)
        return

    def forward(self, signal, context):
        """ 
        Args:
            signal (batchsize, length, signal_size)
            context (batchsize, hidden_dim, length)
        
        Return:
            (batchsize, length, signal_size)
        """
        tmp_h0 = self.l_ff_1(signal) # batchsize, length, hidden_dim)
        tmp_hidden = self.l_ff_1_tanh(tmp_h0).permute(0, 2, 1) # (batchsize, hidden_dim, length)
        
        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        for l_conv in self.l_convs:
            tmp_hidden = tmp_hidden + l_conv(tmp_hidden) + context
            
        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * self.scale

        tmp_hidden = tmp_hidden.permute(0, 2, 1) # (batchsize, length, signal_size)
        
        # compress the dimension and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal
        
        return output_signal
    
# 
# Sine waveform generator
# 
# Sine waveform generator
class SineGen(torch_nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0, 
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SineGen is used inside PulseGen (default False)
    
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    def __init__(self, samp_rate, harmonic_num = 0, 
                 sine_amp = 0.1, noise_std = 0.003,
                 voiced_threshold = 0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
    
    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv
            
    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],\
                              device = f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - 
                                tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) \
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every 
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within 
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines
    
    
    def forward(self, f0):
        """
        Args:
            F0: shape (batchsize, 1, length)
                  f0 for unvoiced steps should be 0

        Return:
            sine_tensor: shape (batchsize, length, dim)
            uv: shape (batchsize, 1, length)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[2], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, 0, :]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx+1] = f0_buf[:, :, 0] * (idx+2)
                
            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp # batchsize, length, dim
            
            # generate uv signal
            uv = self._f02uv(f0) # same shape as f0, (batchsize, 1, length)
            
            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            #.       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1-uv) * self.sine_amp / 3 # (batchsize, 1, length)
            noise = noise_amp[:,0,:][:,:,None] * torch.randn_like(sine_waves) # batchsize, length, dim
            
            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv[:,0,:][:,:,None] + noise # batchsize, length, dim
        return sine_waves, uv, noise

#####
## Model definition
## 

## For condition module only provide Spectral feature to Filter block
class CondModuleHnSincNSF(torch_nn.Module):
    """ Condition module for hn-sinc-NSF
    Upsample and transform input features
    CondModuleHnSincNSF(input_dimension, output_dimension, up_sample_rate,
               blstm_dimension = 64, cnn_kernel_size = 3)
    
    Spec, F0, cut_off_freq = CondModuleHnSincNSF(features, F0)
    Both input features should be frame-level features
    If x doesn't contain F0, just ignore the returned F0
    
    context, f0_upsamp, cut_f_smoothed, hidden_cut_f = 
        CondModuleHnSincNSF(input_dim, output_dim, up_sample, 
                        blstm_s = 64, cnn_kernel_s = 3, 
                        voiced_threshold = 0):
    input_dim: sum of dimensions of input features
    output_dim: dim of the feature Spec to be used by neural filter-block
    up_sample: up sampling rate of input features
    blstm_s: dimension of the features from blstm (default 64)
    cnn_kernel_s: kernel size of CNN in condition module (default 3)
    voiced_threshold: f0 > voiced_threshold is voiced, otherwise unvoiced
    """
    def __init__(self, input_dim, output_dim, up_sample, \
                 blstm_s = 64, cnn_kernel_s = 3, voiced_threshold = 0, \
                 layer_type='default', f0_idx=0, extra_art=False, \
                 elayers=6, dropout=0.2):
        super(CondModuleHnSincNSF, self).__init__()

        # input feature dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.cnn_kernel_s = cnn_kernel_s
        self.cut_f_smooth = up_sample * 4
        self.voiced_threshold = voiced_threshold
        self.f0_idx = f0_idx
        self.extra_art = extra_art

        # condition module
        if layer_type == 'default':
            tmp = []
            if not extra_art:
                tmp += [Conv1dKeepLength(input_dim, output_dim, dilation_s = 1, 
                                        kernel_s = self.cnn_kernel_s)]
            else:
                logging.info("extra_art True not supported")
                sys.exit(1)
                # NOTE need Conv1dChangeLength
            tmp_input_size = [output_dim, output_dim]
            tmp_output_size = [output_dim, output_dim]
            tmp += [Conv1dKeepLength(x, y, dilation_s = 1, 
                                    kernel_s = self.cnn_kernel_s)
                for x, y in zip(tmp_input_size, tmp_output_size)]
            self.l_conv1ds = torch_nn.ModuleList(tmp)
            self.transformer = torch.nn.Identity()
        elif layer_type == 'gblock' or layer_type == 'gblock2':
            tmp = []
            if extra_art:
                tmp += [
                    WNConv1d(input_dim, input_dim*2, kernel_size=2),
                    GBlock(input_dim*2, output_dim, kernel_size=self.cnn_kernel_s, extra_art=extra_art)
                    ]
            else:
                tmp += [GBlock(input_dim, output_dim, kernel_size=self.cnn_kernel_s, extra_art=extra_art)]
            tmp_input_size = [output_dim, output_dim]
            tmp_output_size = [output_dim, output_dim]
            tmp += [GBlock(x, y, kernel_size=self.cnn_kernel_s) for x, y in zip(tmp_input_size, tmp_output_size)]
            self.l_conv1ds = torch_nn.ModuleList(tmp)
            self.transformer = torch.nn.Identity()
        elif layer_type == 'transformer':
            if extra_art:
                self.l_conv1ds = torch_nn.ModuleList([
                    WNConv1d(input_dim, output_dim, kernel_size=2),
                    ResBlock(output_dim, output_dim, 1),
                    ResBlock(output_dim, output_dim, 1),
                    ResBlock(output_dim, output_dim, 1)
                ])
            else:
                self.l_conv1ds = torch_nn.ModuleList([
                    ResBlock(input_dim, output_dim, 1),
                    ResBlock(output_dim, output_dim, 1),
                    ResBlock(output_dim, output_dim, 1)
                ])
            encoder_layer = TransformerEncoderLayer(d_model=output_dim, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=dropout)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, elayers)
            pass
        else:
            logging.info("layer_type not supported")
            sys.exit(1)

        # Upsampling layer for hidden features
        self.l_upsamp = UpSampleLayer(self.output_dim, \
                                      self.up_sample, True)
        # separate layer for up-sampling normalized F0 values
        self.l_upsamp_f0_hi = UpSampleLayer(1, self.up_sample, True)
        
        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = UpSampleLayer(1, self.up_sample, False)

        # Another smoothing layer to smooth the cut-off frequency
        # for sinc filters. Use a larger window to smooth
        self.l_cut_f_smooth = MovingAverage(1, self.cut_f_smooth)

    def get_cut_f(self, hidden_feat, f0):
        """
        
        cut_f = get_cut_f(self, feature, f0)
        
        Args:
            feature: (batchsize, 1, length)
            f0: (batchsize, 1, length)        
        """ 
        # generate uv signal
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        # hidden_feat is between (-1, 1) after conv1d with tanh
        # (-0.2, 0.2) + 0.3 = (0.1, 0.5)
        # voiced:   (0.1, 0.5) + 0.4 = (0.5, 0.9)
        # unvoiced: (0.1, 0.5) = (0.1, 0.5)
        return hidden_feat * 0.2 + uv * 0.4 + 0.3
        
    
    def forward(self, feature, f0):
        """
        
        spec, f0 = forward(self, feature, f0)

        Args:
            feature: (batchsize, in_dim, in_length)
            f0: (batchsize, 1, in_length), which should be F0 at mel/frame-level
        
        Return:
            context: (batchsize, out_dim, out_length)
            f0_upsamp: (batchsize, 1, out_length)
            cut_f_smoothed: (batchsize, 1, out_length)
            hidden_cut_f: (batchsize, 1, out_length)
        """
        tmp = feature # (batchsize, in_dim, in_length)
        for l_conv in self.l_conv1ds:
            tmp = l_conv(tmp) # (batchsize, dim, in_length)
        tmp = tmp.permute(2, 0, 1)  # (in_length, batchsize, dim)
        tmp = self.transformer(tmp) # (in_length, batchsize, dim)
        tmp = tmp.permute(1, 2, 0)
        tmp = self.l_upsamp(tmp) # (batchsize, out_dim, out_length)
        
        if self.extra_art:
            feature = feature[:, :, :-1]
            f0 = f0[:, :, :-1]
        
        # concatenate normed F0 with hidden spectral features
        if self.f0_idx == -1:
            context = torch.cat((tmp[:, 0:self.output_dim-1, :], \
                                self.l_upsamp_f0_hi(feature[:, -1:, :])), \
                                dim=1) # (batchsize, out_dim, out_length)
        else:
            context = torch.cat((tmp[:, 0:self.output_dim-1, :], \
                                self.l_upsamp_f0_hi(feature[:, self.f0_idx:self.f0_idx+1, :])), \
                                dim=1) # (batchsize, out_dim, out_length)

        # hidden feature for cut-off frequency
        hidden_cut_f = tmp[:, self.output_dim-1:, :] # (batchsize, 1, out_length)

        # directly up-sample F0 without smoothing
        f0_upsamp = self.l_upsamp_F0(f0) # (batchsize, 1, out_length)

        # get the cut-off-frequency from output of CNN
        cut_f = self.get_cut_f(hidden_cut_f, f0_upsamp) # (batchsize, 1, out_length)
        # smooth the cut-off-frequency using fixed average smoothing
        cut_f_smoothed = self.l_cut_f_smooth(cut_f) # (batchsize, 1, out_length)

        return context, f0_upsamp, cut_f_smoothed, hidden_cut_f

# For source module
class SourceModuleHnNSF(torch_nn.Module):
    """ SourceModule for hn-nsf 
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshold=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threshold to set U/V given F0 (default: 0)
    """
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshold=0):
        super(SourceModuleHnNSF, self).__init__()
        
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshold)

        # to merge source harmonics into a single excitation
        self.l_linear = torch_nn.Linear(harmonic_num+1, 1)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: F0_upsampled, shape (batchsize, 1, length)
        
        Return:
            sine_merge: Sine_source, shape (batchsize, length, 1)
            noise: noise_source, shape (batchsize, length, 1)
            uv: shape (batchsize, 1, length)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
            # sine_wavs: shape (batchsize, length, dim)
            # uv: shape (batchsize, 1, length)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise.permute(0, 2, 1), uv
        
        
# For Filter module
class FilterModuleHnSincNSF(torch_nn.Module):
    """ Filter for Hn-sinc-NSF
    FilterModuleHnSincNSF(signal_size, hidden_size, sinc_order = 31,
                          block_num = 5, kernel_size = 3, 
                          conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    sinc_order: order of the sinc filter
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block
    Usage:
    output = FilterModuleHnSincNSF(har_source, noi_source, context, cut_f)
    har_source: source for harmonic branch (batchsize, length, dim=1)
    noi_source: source for noise branch (batchsize, length, dim=1)
    cut_f: cut-off-frequency of sinc filters (batchsize, length, dim=1)
    context: hidden features to be added (batchsize, length, dim)
    output: (batchsize, length, dim=1)    
    """
    def __init__(self, signal_size, hidden_size, sinc_order = 31, \
                 block_num = 5, kernel_size = 3, conv_num_in_block = 10, layer_type='default'):
        super(FilterModuleHnSincNSF, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block
        self.sinc_order = sinc_order
        
        # filter blocks for harmonic branch
        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block, layer_type=layer_type) \
               for x in range(self.block_num)]
        self.l_har_blocks = torch_nn.ModuleList(tmp)

        # filter blocks for noise branch (only one block, 5 sub-blocks)
        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block // 2, layer_type=layer_type) \
               for x in range(1)]
        self.l_noi_blocks = torch_nn.ModuleList(tmp)

        # sinc filter generators and time-variant filtering layer
        self.l_sinc_coef = SincFilter(self.sinc_order)
        self.l_tv_filtering = TimeVarFIRFilter()
        # done
        

    def forward(self, har_component, noi_component, cond_feat, cut_f):
        """
        Args:
            har_component/har_source: source for harmonic branch (batchsize, length, 1)
            noi_component/noi_source: source for noise branch (batchsize, length, 1)
            cond_feat/context: hidden features to be added (batchsize, hidden_dim, length)
            cut_f: cut-off-frequency of sinc filters (batchsize, 1, length)
        
        Return:
            output: (batchsize, length, 1)
        """
        # eg (cond_feat.shape) # 16, 8140, 64
        # harmonic component
        for l_har_block in self.l_har_blocks:
            har_component = l_har_block(har_component, cond_feat) # (batchsize, length, 1)
        # noise component
        for l_noi_block in self.l_noi_blocks:
            noi_component = l_noi_block(noi_component, cond_feat) # (batchsize, length, 1)
        
        # get sinc filter coefficients
        lp_coef, hp_coef = self.l_sinc_coef(cut_f.permute(0, 2, 1))

        # time-variant filtering
        har_signal = self.l_tv_filtering(har_component, lp_coef)
        noi_signal = self.l_tv_filtering(noi_component, hp_coef)

        # get output 
        return har_signal + noi_signal
        
        

## FOR MODEL
class NSFA2WModel(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_channels=80, out_channels=1, sampling_rate=44100, hop_size=110, mean_std=None,
                    hidden_dim=64, filter_block_num=5, layer_type="default", f0_idx=0, extra_art=False, 
                    use_ar=False, ar_type="fc", ar_input=512, ar_hidden=256, ar_output=128, 
                    art_len=100, ar_dropout=0.2, ar_elayers=2, ar_ffdim=512, elayers=6):
        super(NSFA2WModel, self).__init__()

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_channels,out_channels, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        self.input_dim = in_channels
        self.output_dim = out_channels
        self.f0_idx = f0_idx
        self.extra_art = extra_art
        self.use_ar = use_ar
        self.ar_type = ar_type

        # configurations
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = 0.1
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = 0.003
        # dimension of hidden features in filter blocks
        self.hidden_dim = hidden_dim
        # upsampling rate on input acoustic features (16kHz * 5ms = 80)
        # assume input_reso has the same value
        self.upsamp_rate = hop_size
        # sampling rate (Hz)
        self.sampling_rate = sampling_rate
        # CNN kernel size in filter blocks        
        self.cnn_kernel_s = 3
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = filter_block_num
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = 10
        # number of harmonic overtones in source
        self.harmonic_num = 7
        # order of sinc-windowed-FIR-filter
        self.sinc_order = 31

        # the three modules
        self.m_cond = CondModuleHnSincNSF(self.input_dim, \
                                          self.hidden_dim, \
                                          self.upsamp_rate, \
                                          cnn_kernel_s=self.cnn_kernel_s, \
                                          layer_type=layer_type, f0_idx=f0_idx, extra_art=extra_art, \
                                          elayers=elayers)

        self.m_source = SourceModuleHnNSF(self.sampling_rate, 
                                          self.harmonic_num, 
                                          self.sine_amp, self.noise_std)
        
        self.m_filter = FilterModuleHnSincNSF(self.output_dim, \
                                              self.hidden_dim, \
                                              self.sinc_order, \
                                              self.filter_block_num, \
                                              self.cnn_kernel_s, \
                                              self.cnn_num_in_block, layer_type=layer_type)
        
        if self.use_ar:
            if self.ar_type == "fc":
                self.ar_model = PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)
            else:
                self.ar_model = PastSeqEncoder(output_dim=ar_output, dropout=ar_dropout, elayers=ar_elayers, ffdim=ar_ffdim)
                self.interp_fn = functools.partial(
                    torch.nn.functional.interpolate,
                    size=art_len,
                    mode='linear',
                    align_corners=False)

        return
    
    def prepare_mean_std(self, in_dim, out_dim, data_mean_std=None):
        """
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        """
        return (x - self.input_mean[None, :, None]) / self.input_std[None, :, None]

    def normalize_target(self, y):
        """ normalizing the target data
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        """
        return y * self.output_std + self.output_mean
    
    def forward(self, x, ar=None):
        """ definition of forward method 
        
        Args:
            x: shape (batchsize, dim, art_length), eg (16, 30, 74)
            ar: shape (batch_size, 1, audio_length)
        """
        if self.use_ar:
            ar_feats = self.ar_model(ar)
            if self.ar_type == "fc":
                # ar_feats (batchsize, ar_output)
                ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, x.shape[2]) # (batchsize, ar_output, length)
            else:
                # ar_feats (batch_size, ar_output, ar_len)
                ar_feats = self.interp_fn(ar_feats)
            x = torch.cat((x, ar_feats), dim=1)
        # t0 = time.time()
        if self.f0_idx == -1:
            f0 = x[:, -1:, :] # (batchsize, 1, length)
        else:
            f0 = x[:, self.f0_idx:self.f0_idx+1, :] # (batchsize, 1, length)
        # normalize the input features data
        feat = self.normalize_input(x) # (batchsize, dim, length)
        # t1 = time.time()
        # logging.info('%.4f' % (t1-t0))

        # condition module
        # feature-to-filter-block, f0-up-sampled, cut-off-f-for-sinc,
        #  hidden-feature-for-cut-off-f
        cond_feat, f0_upsamped, cut_f, hid_cut_f = self.m_cond(feat, f0)
            # cond_feat: (batchsize, hidden_dim, out_length)
            # f0_upsamped: (batchsize, 1, out_length)
            # cut_f: (batchsize, 1, out_length)
        # t1 = time.time()
        # logging.info('%.4f' % (t1-t0))

        # source module
        # harmonic-source, noise-source (for noise branch), uv
        har_source, noi_source, uv = self.m_source(f0_upsamped)
            # har_source: (batchsize, length, 1)
            # noi_source: (batchsize, length, 1)
        # t1 = time.time()
        # logging.info('%.4f' % (t1-t0))
        
        # neural filter module (including sinc-based FIR filtering)
        # output
        output = self.m_filter(har_source, noi_source, cond_feat, cut_f)
            # (batchsize, length, 1)
        output = output.permute(0, 2, 1) # (batchsize, 1, length)
        # t1 = time.time()
        # logging.info('%.4f' % (t1-t0))
        return output
        '''
        if self.training:
            # just in case we need to penalize the hidden feauture for 
            # cut-off-freq. 
            return [output.squeeze(-1), hid_cut_f]
        else:
            return output.squeeze(-1)
        '''
    
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
        x = x.permute(0, 2, 1)
        return self.forward(x)

    
if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    model = NSFA2WModel(**generator_params)
    x = torch.ones(1, 30, 17)
    t0 = time.time()
    with torch.no_grad():
        o = model(x)
    t1 = time.time()
    print(t1-t0)
    '''
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )
    '''
