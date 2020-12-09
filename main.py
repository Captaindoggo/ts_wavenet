from data import LJ_speech, my_collate, MelSpectrogram, MelSpectrogramConfig
from model import WaveNet, Block, count_parameters

from os import listdir
from os.path import isfile, join

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import string
import numpy as np
from math import isnan
import random

import os
import glob
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from torch.utils.data import random_split

from tqdm import tqdm_notebook
from torch.optim import Adam

from IPython import display
from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa
from matplotlib import pyplot as plt

def set_seed(n):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed = n
    random.seed(n)
    np.random.seed(n)

if __name__ == '__main__':

    set_seed(42)

    featurizer = MelSpectrogram(MelSpectrogramConfig())

    mu_law = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
    mu_law_dec = torchaudio.transforms.MuLawDecoding(quantization_channels=256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveNet(device).to(device)

    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1, factor=0.5)

    criterion = nn.CrossEntropyLoss()

    CROP = 12800
    # root = '/content/drive/MyDrive/DLA/hw5/LJSpeech-1.1/wavs/'

    root = 'LJSpeech-1.1/wavs/'

    wav_ids = [f for f in listdir(root) if isfile(join(root, f))]

    dataset = LJ_speech(wav_ids, root, crop=CROP)

    data_len = len(dataset)
    test_len = int(data_len * 0.2)
    train_len = data_len - test_len

    train, val = random_split(dataset, [train_len, test_len])

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               collate_fn=partial(my_collate, mu_law=mu_law, featurizer=featurizer))

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True,
                                             collate_fn=partial(my_collate, mu_law=mu_law, featurizer=featurizer))

    epochs = 8
    pbar = tqdm_notebook(total=epochs * (len(train) // batch_size))

    for epoch in range(0, epochs):
        running_loss = 0.0
        val_loss = 0.0
        ctr = 0
        val_ctr = 0
        model.train()
        for data in train_loader:
            mel, wav = data
            mel = mel.to(device)
            wav = wav.to(device)
            optimizer.zero_grad()
            output = model(mel, wav[:, :-1])
            output = output.transpose(-1, -2)

            loss = criterion(output.contiguous().view(-1, 256),
                             wav[:, 1:].contiguous().view(-1).type(torch.LongTensor).to(device))
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            ctr += 1
            pbar.update(1)

        torch.save(model.state_dict(), 'weights'+str(epoch) + '.pt')

        for data in val_loader:
            mel, wav = data
            mel = mel.to(device)
            wav = wav.to(device)
            model.eval()
            with torch.no_grad():
                output = model(mel, wav[:, :-1])
                output = output.transpose(-1, -2)

                loss = criterion(output.contiguous().view(-1, 256),
                                 wav[:, 1:].contiguous().view(-1).type(torch.LongTensor).to(device))
                val_loss += loss.item()

            val_ctr += 1
        print('epoch:', epoch + 1, 'train loss:', running_loss / ctr, 'val loss:', val_loss / val_ctr)
        scheduler.step(val_loss)
    pbar.close()