import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.io.wavfile import read
from torch.utils.data import Dataset, DataLoader
from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork

import os
import sys
import glob
import argparse
from tqdm import tqdm


class AudioDataSet:
    def __init__(self, data_dir):
        print("Loading data")
        dir = os.listdir(data_dir)
        x = np.zeros((len(dir), 1, 16384))
        i = 0
        for file in tqdm(dir):
            audio = read(data_dir+file)[1]
            if audio.shape[0] < 16384:
                audio = np.pad(audio, (0, 16384-audio.shape[0]))
            audio = audio[:16384]
            x[i, 0, :] = audio
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x/32767, dtype=np.float32))


    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return self.len


def gradient_penalty(G, D, real, fake, epsilon):
    x_hat = epsilon * real + (1 - epsilon) * fake
    scores = D(x_hat)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1) # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty

if __name__ == "__main__":
    # TODO: Add more args
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_dir', type=str, help='Training Directory')

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./training/"
    log_dir = './checkpoints/'
    WAVEGAN_DISC_NUPDATES = 5
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9

    # Load data
    dataset = AudioDataSet(data_dir)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        drop_last=True
    )

    # Load models
    G = WaveGANGenerator().to(device).train()
    D = WaveGANDiscriminator().to(device).train()

    # Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))


    for epoch in range(NUM_EPOCHS):
        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        pbar = tqdm(dataloader)
        real = dataset[:BATCH_SIZE].to(device)
        for i, real in enumerate(pbar):
            # D Update
            optimizer_D.zero_grad()
            real = real.to(device)
            epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, 16384).to(device)
            z = torch.FloatTensor(BATCH_SIZE, 100).uniform_(-1, 1).to(device)
            fake = G(z)
            penalty = gradient_penalty(G, D, real, fake, epsilon)

            D_loss = torch.mean(D(fake) - D(real) + LAMBDA * penalty)
            D_loss.backward()
            optimizer_D.step()

            # G Update
            if i % WAVEGAN_DISC_NUPDATES == 0:
                optimizer_G.zero_grad()
                z = torch.FloatTensor(BATCH_SIZE, 100).uniform_(-1, 1).to(device)
                G_loss = torch.mean(-D(G(z)))
                G_loss.backward()
                optimizer_G.step()
        torch.save(G.state_dict(), f'./checkpoints/epoch{epoch}_G.pt')
        torch.save(G.state_dict(), f'./checkpoints/epoch{epoch}_D.pt')
