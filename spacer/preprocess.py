
import os
import pickle

from config import ARGS


def encode_string(s, chr_to_idx):
    encoded = []
    for c in s:
        try:
            idx = chr_to_idx[c]
        except:
            idx = chr_to_idx['<UNK>']

        encoded.append(idx)

    return encoded

def get_label(s):
    label = []
    checker = False
    for i in range(len(s)):
        if s[i] == " ":
          checker = True
        elif checker == True:
          label.append(1)
          checker = False
        else:
          label.append(0)
    return label

def prepare_sentence(train_data, vocab_dic):

    with open(train_data, 'r', encoding="utf-8") as f:

        b_texts = f.readlines()

        print('reading complete..')

        #remove redundent whitespace and change uppercase to lowercase
        texts = [' '.join(t.split()).lower() for t in b_texts]

        # Y data 
        labels = [get_label(t) for t in texts]

        # Remove whitespaces.
        texts = [''.join(t.split()) for t in texts]

        print('start encoding... ') 
        texts = [encode_string(t, vocab_dic) for t in texts]
        print('end encoding...')

    return texts, labels