# Articulation GAN: Unsupervised Modeling of Articulatory Learning
> A PyTorch Implementation for [Articulation GAN](https://arxiv.org/pdf/2210.15173.pdf)

## Setup
```bash
git clone https://github.com/gbegus/articulationGAN.git
cd articulationGAN
pip install -r requirements.txt
```
Here is a [link](https://drive.google.com/drive/folders/1_Sfjm0NhDAw6Zy3QnxnJY01DUwl24OUc?usp=sharing) to a folder containing the weights of several ema2wav models.

## Training
```bash
cd articulationGAN
python train.py --datadir data_dir/ --logdir log_dir/ --emadir articulatory_weights/ --ciw
```
Here is a list of the possible command line options for training:

| Argument | Description |
| -------- | ---------- |
|datadir | Path to a folder containing .wav files for training the model |
|logdir | Path to the folder where checkpoints and training logs will be stored |
|emadir | Path to a folder containing the weights of the ema2wav model |
|slice_len | Slice length of training samples. Shorter samples will be zero-padded and longer samples will be cropped to the specified length. The provided ema2wav models only support the default slice_len of 20480.|
|kernel_len| Kernel length of the ArticulationGAN generator. Must be an odd integer; the suggested range is from 3 to 25|
|num_channels | Possible values: **12** or **13** The number of EMA channels that the model will generate. The provided folder contains ema2wav models supporting 12 and 13 channels, which will be automatically loaded based on the value of this argument.
|log_audio | If used, this flag will allow the trainer to log sample audio files and EMA plots periodically. Otherwise, only the losses will be saved in the training log. This may increase the filesize for longer runs. |
|num_categ | The number of categories used for Q-network training. This should be equivalent to the number of classes in the training dataset.|
|**ciw** or **fiw**| Mutually exclusive arguments that determine whether categorical (ciw) or featural (fiw) z-vectors will be used in the generator. One of the two is required to enable learning using the Q network. ciw is generally recommended for most training runs. More information can be found in [this](https://www.sciencedirect.com/science/article/pii/S0893608021001052?via%3Dihub#b28) paper.
|save_int| Save interval in epochs|
|batch_size| Batch size| 
|cont| Provide the epoch number to resume training from a specific checkpoint, or set to "last" to continue from the last available checkpoint.|
