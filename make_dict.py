#-*- coding: utf-8 -*-
import os

from config import ARGS
import pickle

def make_dic(train_data, valid_data):
    X_data = []
    Y_data = []

    with open(train_data, 'r', encoding="utf-8") as t:
        X_data = t.readlines()
        print(len(X_data))

    print(len(X_data))
    print('reading complete.. ')

    X_texts = [' '.join(t.split()).lower() for t in X_data]
    X_texts = [''.join(t.split()) for t in X_texts]

    vocab = ['<pad>'] + ['<UNK>'] + sorted(set([char for seq in X_texts for char in seq]))

    return vocab

if __name__ == '__main__':

    vocab = make_dic(ARGS.train_data, ARGS.val_data)

    print('사전 단어 개수 ', len(vocab))
    print(vocab[0], vocab[1])
    
    with open('aihub_dic.pickle', 'wb') as f:
      pickle.dump(vocab, f)