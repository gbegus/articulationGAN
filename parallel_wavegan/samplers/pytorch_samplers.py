import json
import logging
import numpy as np
import os
import random
import string
import torch


class SizeAwareSampler(torch.utils.data.Sampler):
    '''
    from David Gaddy's
    https://github.com/dgaddy/subvocal/blob/master/read_emg.py
    '''
    def __init__(self, audio_lens, max_len=2000):
        self.audio_lens = audio_lens
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.audio_lens)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            length = self.audio_lens[idx]
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch
