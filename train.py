import argparse
import yaml
import os
import re

import numpy as np
import torch
import torch.optim as optim
from scipy.io.wavfile import read
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from articulatory.utils import load_model
from tqdm import tqdm

from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork
from utils import get_continuation_fname


class AudioDataSet:
    def __init__(self, datadir, slice_len):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        i = 0
        for file in tqdm(dir):
            audio = read(os.path.join(datadir, file))[1]
            if audio.shape[0] < slice_len:
                audio = np.pad(audio, (0, slice_len - audio.shape[0]))
            audio = audio[:slice_len]

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
            audio /= np.max(np.abs(audio))
            x[i, 0, :] = audio
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))

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
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty

def synthesize(model, x, config):
    '''
    Given batch of EMA data and EMA model, synthesizes speech output
    Args:
        x: (batch, art_len, num_feats)

    Return:
        signal: (batch, audio_len)
    '''
    batch_size = x.shape[0]
    params_key = "generator_params"
    audio_chunk_len = config["batch_max_steps"]
    in_chunk_len = int(audio_chunk_len/config["hop_size"])
    past_out_len = config[params_key]["ar_input"]

    # NOTE extra_art not supported
    ins = [x[:, i:i+in_chunk_len, :] for i in range(0, x.shape[1], in_chunk_len)]
    prev_samples = torch.zeros((batch_size, config[params_key]["out_channels"], past_out_len), dtype=x.dtype, device=x.device)
    outs = []

    for cin in ins: # a2w cin (batch_size, in_chunk_len, num_feats)
        cin = cin.permute(0, 2, 1)  # a2w (batch_size, num_feats, in_chunk_len)
        cout = model(cin, ar=prev_samples)  # a2w (batch_size, 1, audio_chunk_length)
        outs.append(cout[:, 0, :])
        if past_out_len <= audio_chunk_len:
            prev_samples = cout[:, :, -past_out_len:]
        else:
            prev_samples[:, :, :-in_chunk_len] = prev_samples[:, :, in_chunk_len:].clone()
            prev_samples[:, :, -in_chunk_len:] = cout
    out = torch.unsqueeze(torch.cat(outs, dim=1), 1)  # w2a (batch_size, seq_len, num_feats)
    return out

if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Log/Results Directory'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=1,
        help='Q-net categories'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=20480,
        help='Length of training data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--cont',
        type=str,
        default="",
        help='''continue: default from the last saved iteration. '''
             '''Provide the epoch number if you wish to resume from a specific point'''
             '''Or set "last" to continue from last available'''
    )

    parser.add_argument(
        '--save_int',
        type=int,
        default=50,
        help='Save interval in epochs'
    )

    # Q-net Arguments
    Q_group = parser.add_mutually_exclusive_group()
    Q_group.add_argument(
        '--ciw',
        action='store_true',
        help='Trains a ciwgan'
    )
    Q_group.add_argument(
        '--fiw',
        action='store_true',
        help='Trains a fiwgan'
    )
    args = parser.parse_args()
    train_Q = args.ciw or args.fiw

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthesis_checkpoint_path = "articulatory_checkpoints/mocha_train_lcdx0pmf8nema_mocha2w_hifi_lcdx0pm/best_mel_ckpt.pkl"
    synthesis_config_path = "articulatory_checkpoints/mocha_train_lcdx0pmf8nema_mocha2w_hifi_lcdx0pm/config.yml"
    with open(synthesis_config_path) as f:
        synthesis_config = yaml.load(f, Loader=yaml.Loader)
    datadir = args.datadir
    logdir = args.logdir
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    BATCH_SIZE = args.batch_size
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9

    CONT = args.cont
    SAVE_INT = args.save_int

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )


    def make_new():
        G = WaveGANGenerator(nch=40).to(device).train()
        EMA = load_model(synthesis_checkpoint_path, synthesis_config)
        EMA.remove_weight_norm()
        EMA = EMA.eval().to(device)
        D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        Q, optimizer_Q, criterion_Q = (None, None, None)
        if train_Q:
            Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
        if args.fiw:
            optimizer_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)
            criterion_Q = torch.nn.BCEWithLogitsLoss()
        elif args.ciw:
            optimizer_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)
            criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])

        return G, D, EMA, optimizer_G, optimizer_D, Q, optimizer_Q, criterion_Q


    # Load models
    G, D, EMA, optimizer_G, optimizer_D, Q, optimizer_Q, criterion_Q = make_new()
    start_epoch = 0
    start_step = 0

    if CONT.lower() != "":
        try:
            print("Loading model from existing checkpoints...")
            fname, start_epoch = get_continuation_fname(CONT, logdir)

            G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_G.pt")))
            D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_D.pt")))
            if train_Q:
                Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q.pt")))

            optimizer_G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Gopt.pt")))
            optimizer_D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Dopt.pt")))

            if train_Q:
                optimizer_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Qopt.pt")))

            start_step = int(re.search(r'_step(\d+).*', fname).group(1))
            print(f"Successfully loaded model. Continuing training from epoch {start_epoch},"
                  f" step {start_step}")

        # Don't care why it failed
        except Exception as e:
            print("Could not load from existing checkpoint, initializing new model...")
            print(e)
    else:
        print("Starting a new training")

    # Set Up Writer
    writer = SummaryWriter(logdir)
    step = start_step

    for epoch in range(start_epoch + 1, NUM_EPOCHS):

        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        pbar = tqdm(dataloader)
        real = dataset[:BATCH_SIZE].to(device)

        for i, real in enumerate(pbar):
            # D Update
            optimizer_D.zero_grad()
            real = real.to(device)
            epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
            _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
            if train_Q:
                if args.fiw:
                    c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                else:
                    c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                    num_classes=NUM_CATEG).to(device)
                z = torch.cat((c, _z), dim=1)
            else:
                z = _z

            fake = synthesize(EMA, G(z).permute(0, 2, 1), synthesis_config)
            penalty = gradient_penalty(G, D, real, fake, epsilon)

            D_loss = torch.mean(D(fake) - D(real) + LAMBDA * penalty)
            writer.add_scalar('Loss/Discriminator', D_loss.detach().item(), step)
            D_loss.backward()
            optimizer_D.step()

            if i % WAVEGAN_DISC_NUPDATES == 0:
                optimizer_G.zero_grad()
                EMA.zero_grad()
                if train_Q:
                    optimizer_Q.zero_grad()
                _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)

                if train_Q:
                    if args.fiw:
                        c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                    else:
                        c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                        num_classes=NUM_CATEG).to(device)

                    z = torch.cat((c, _z), dim=1)
                else:
                    z = _z
                G_z = synthesize(EMA, G(z).permute(0, 2, 1), synthesis_config)

                # G Loss
                G_loss = torch.mean(-D(G_z))
                G_loss.backward(retain_graph=True)
                writer.add_scalar('Loss/Generator', G_loss.detach().item(), step)

                # Q Loss
                if train_Q:
                    Q_loss = criterion_Q(Q(G_z), c)
                    Q_loss.backward()
                    writer.add_scalar('Loss/Q_Network', Q_loss.detach().item(), step)
                    optimizer_Q.step()

                # Update
                optimizer_G.step()
            step += 1

        if not epoch % SAVE_INT:
            torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))
            if train_Q:
                torch.save(Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q.pt'))

            torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            if train_Q:
                torch.save(optimizer_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Qopt.pt'))
