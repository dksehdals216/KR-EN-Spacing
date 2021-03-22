
import os
import torch
import pickle
import torch.nn as nn

from config import ARGS
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor

torch.manual_seed(1)

def int_to_string(org_data, bin_data, vocab):

    sents = []

    for sentence, sent2 in zip(org_data, bin_data):
        sen = ""
        for char, bin in zip(sentence, sent2):
            if vocab[char] == '<pad>':
                break
            else:
                sen += (vocab[char] if bin==0 else ' '+vocab[char])
        sents.append(sen)

    return sents

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, train_x, train_y):
        self.x_data = train_x
        self.y_data = train_y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y

def collate_fn(batch):
  (xx, yy) = zip(*batch)

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).cuda()
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0).cuda()

  return xx_pad, yy_pad
