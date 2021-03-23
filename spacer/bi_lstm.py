#-*- coding: utf-8 -*-
# https://github.com/jidasheng/bi-lstm-crf/blob/master/bi_lstm_crf/app/train.py
# 모델

import argparse
import neptune
import random
import os
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from collections import OrderedDict
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

from config import ARGS


torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)

if use_cuda == 1:
  print('cuda is available')

best_acc = 0

class third_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(third_LSTM, self).__init__()
        self.hidden_dim =  hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Forward LSTM
        self.forward_lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Backward LSTM 
        self.backward_lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim * 2, 1)
        self.hidden_forward = self.init_hidden()
        self.hidden_backward = self.init_hidden()
        self.vocab_size = vocab_size

    def init_hidden(self):
        return (torch.zeros(1, ARGS.batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, ARGS.batch_size, self.hidden_dim).cuda())

    def flipData(self, data, lengths):
        for i in range(data.shape[0]):
          data[i,:int(lengths[i]),:] = data[i,:int(lengths[i]),:].flip(dims=[0])
        return data

    def forward(self, x, mask):
        Bn, Tx = x.size()

        #Calculate each sentence length
        length = mask.sum(dim=1)

        # Embedding process (Batch size, Max sequence Length, Embedding dimmension)
        embeds = self.word_embeddings(x)

        #Forward
        packed_forward = torch.nn.utils.rnn.pack_padded_sequence(embeds, length, batch_first = True, enforce_sorted = False)
        output_forward, _ = self.forward_lstm(packed_forward, self.hidden_forward)
        unpacked_forward = torch.nn.utils.rnn.pad_packed_sequence(output_forward, batch_first=True)[0]

        reversed_embeds = self.flipData(embeds, length)

        #Backward
        packed_backward = torch.nn.utils.rnn.pack_padded_sequence(reversed_embeds, length, batch_first = True, enforce_sorted = False)
        output_backward, _ = self.backward_lstm(packed_backward, self.hidden_backward)
        unpacked_backward = torch.nn.utils.rnn.pad_packed_sequence(output_backward, batch_first=True)[0]

        unpacked_backward = self.flipData(unpacked_backward,length)
        merged = torch.cat((unpacked_forward, unpacked_backward), 2)

        #To pass to linear, Flatten the result from lstm
        merged = merged.contiguous()
        merged = merged.view(-1, merged.shape[2])
        tag_space = self.hidden2tag(merged)
        tag_scores = torch.sigmoid(tag_space)

        return tag_scores.view(Bn, -1, 1)

class MY_RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
      super(MY_RNN, self).__init__()
      self.hidden_dim =  hidden_dim
      self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
      # Linear Layer: hidden_dim -> target_size(0: non_spacing, 1: spacing)
      self.hidden2tag = nn.Linear(hidden_dim*2, 1)
      self.hidden = self.init_hidden()
      self.vocab_size = vocab_size
      self.dropout = nn.Dropout(0.3)

      self.hidden2tag2 = nn.Linear(48, 1)
      self.dropout2 = nn.Dropout(0.3)
      
    def init_hidden(self):  
        # LSTM은 hidden state, output 2개가 필요하다. torch.zeros(num_layers, batch_size, hidden_dim)
        return (torch.zeros(2, ARGS.batch_size, self.hidden_dim).cuda(),
                torch.zeros(2, ARGS.batch_size, self.hidden_dim).cuda())
    
    def forward(self, x, mask):
        Bn, Tx = x.size()
        #Calculate each sentence length
        length = mask.sum(dim=1)

        # Embedding process (Batch size, Max sequence Length, Embedding dimmension)
        embeds = self.word_embeddings(x)

        packed_x = torch.nn.utils.rnn.pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        output_x, _ = self.lstm(packed_x, self.hidden)
        unpacked_x = torch.nn.utils.rnn.pad_packed_sequence(output_x, batch_first=True)[0]

        #To pass to linear, Flatten the result from lstm
        output = unpacked_x.contiguous()  

        tag_space = self.hidden2tag(output)

        return tag_space

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
        if i==0:
          label.append(1)
        elif s[i] == " ":
          checker = True
        elif checker == True:
          label.append(1)
          checker = False
        else:
          label.append(0)
    return label

def prepare_sentence(train_data, vocab_dic):

    with open(train_data, 'r', encoding="utf-8") as f:
        #FIle reading
        b_texts = f.readlines()

        print('reading complete.. and shuffle')
        random.shuffle(b_texts) # shuffle 해주기.

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

def int_to_string(org_data, bin_data, vocab):
  # 0:<pad>인 이유 0이 패드인지 원래 문자를 인코딩한 숫자인지 구분할 수 없을 수도 있으니까.
  sents = ""
  for char, bin in zip(org_data, bin_data):

    if vocab[char] == '<pad>':
      break
    else:
      sents += (vocab[char] if bin==0 else ' '+vocab[char])
  
  return sents

def bcelog_to_binary(outputs, mask):
  lists = []
  outputs = torch.sigmoid(outputs)

  for tag in outputs:
    li = []
    for part_tag in tag:
      li.append(1 if part_tag>=0.5 else 0)
    lists.append(li)

  return torch.tensor(lists)

def cal_acc(outputs, y, mask, sent_enc, valid_vocab):
    tot_acc = 0
    for i, (Os, Ys, Ms, Es) in enumerate(zip(outputs.to(device), y.to(device), mask, sent_enc)):

      Ms = Ms.gt(0)

      Os = Os.masked_select(Ms)
      Ys = Ys.masked_select(Ms)
      sent_leng = len(Ys)

      num_corrects = (Os.to(device=device, dtype=torch.int64) == Ys.to(device=device, dtype=torch.int64)).sum().item()

      tot_acc += num_corrects/sent_leng
      spaced = int_to_string(Es, Os, valid_vocab)    # 마지막 문장 반환

    return tot_acc/ARGS.batch_size, spaced, Os

def validate(model, loss_func, loader_valid, valid_vocab):
  total_loss, total_acc = 0, 0
  global best_acc

  model.eval()

  with torch.no_grad():
    for i, (sent_enc, y) in enumerate(loader_valid):
      # forward
      mask = sent_enc.ne(0).float()
      y_out = model(sent_enc, mask) # tag_score의 shape: (batch_Size, max_seq, 1), sigmoid를 통과시킨 값을 가짐. 
      mask = mask.gt(0)

      # calculate loss
      loss_val = loss_cal(y_out, y, mask, loss_func)

      # total loss & acc
      total_loss += loss_val

      outputs = bcelog_to_binary(y_out, mask)
      
      t_acc, spaced, O = cal_acc(outputs, y, mask, sent_enc, valid_vocab) # Mini-batch 1개당 문장의 평균 정확도
      total_acc += t_acc
      
      if ARGS.isneptune == 1:
          # neptune.log_metric('val_acc', acc)
          neptune.log_metric('every_val_loss', loss_val)
      if (i+1) % ARGS.valid_print == 0:
        loss = (total_loss/ARGS.valid_print)# 한문장에 대한 평균 로스값
        acc = total_acc/ARGS.valid_print # 한문장에 대한 평균 정확도
        print('[Validation loss] {l:.4f} \n[Spaced] {s}\n'.format(l=loss, s=spaced))

        total_loss = 0
        total_acc = 0
        if ARGS.isneptune == 1:
          # neptune.log_metric('val_acc', acc)
          neptune.log_metric('val_loss', loss)
        if acc > best_acc:
          best_acc = acc
          print('[Best ACC] {a:.4f}\n'.format(a=best_acc))

          # 모델 저장
          cur_weight = model.state_dict()
          torch.save(cur_weight, '{}.pt'.format(ARGS.model_name))

  model.train()

def loss_cal(pred_y, y, mask, loss_function):

    loss = loss_function(pred_y.squeeze(), y)
    loss = loss.masked_select(mask).mean()

    return loss

def LSTM_train(model, loader_train, loader_valid, vocab_dic):
    global best_acc
    num_corrects, num_total, count, epoch = 0, 0, 0, 0

    loss_total = 0
    acc_total, f1_total = 0, 0

    min_val_loss = np.Inf

    if ARGS.isneptune == 1:
        neptune.init(project_qualified_name='timothy/sandbox', 
              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZDI5OGE4NWItMmE1OS00ZWUzLTkwMmUtOTU4Y2Y1YmRhYjc2In0=',
            )
        neptune.create_experiment(name=ARGS.model_name)
    
    pos_W = torch.FloatTensor([1.3])
    if ARGS.loss_func == "BCELoss":
      loss_function = nn.BCELoss(reduction="none")
    elif ARGS.loss_func == "NLLLoss":
      loss_function = nn.NLLLoss(reduction="none")
    elif ARGS.loss_func == "BCEWithLogitsLoss":
      print(ARGS.loss_func)
      loss_function = nn.BCEWithLogitsLoss(reduction="none")#, pos_weight=pos_W)

    if ARGS.optim == "Adam":
      optimizer = optim.Adam(model.parameters(), lr=ARGS.learning_rate)
    elif ARGS.optim == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=ARGS.learning_rate)

    model.to(device)
    loss_function.to(device)

    while True: # early stopping
        epoch += 1
        model.train()

        for i, (sentence, y) in enumerate(loader_train):
            model.zero_grad()


            mask = sentence.ne(0)
            tag_score = model(sentence, mask) # forwarding

            train_loss = loss_cal(tag_score, y, mask, loss_function)
            loss_total += train_loss

            outputs = bcelog_to_binary(tag_score, mask)            
            t_acc, spaced, O = cal_acc(outputs, y, mask, sentence, vocab_dic) 

            # Perform backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            report = classification_report(y[0].cpu(), outputs[0], target_names=['non_spacing', 'spacing'], output_dict=True)
            f1s = report['spacing']['f1-score']
            f1_total += f1s
            acc_total += t_acc

            if (i+1) % ARGS.train_print == 0:
                
                train_loss_avg = loss_total/(ARGS.train_print)
                acc_avg = acc_total/(ARGS.train_print)
                
                print('[Epoch]: {e}, [Loss]: {l:.6f}, [ACC]: {a:.4f}, [F1]: {f:.4f}, {i} is trained.... '.format(
                  e=epoch, l=train_loss_avg, a=acc_avg, f=f1_total/(ARGS.train_print), i=i+1))
                print('[Expected spacing] {o}\n'.format(o=spaced))
                # print(report)
                
                cur_weight = model.state_dict()
                torch.save(cur_weight, ARGS.model_name)
                loss_total, acc_total, f1_total = 0, 0, 0

                if ARGS.isneptune == 1:
                  neptune.log_metric('train_loss', train_loss_avg)

            # Validation
            if (i+1)>=ARGS.valid and (i+1)%ARGS.is_validation == 0:

                val_loss_total = validate(model, loss_function, loader_valid, vocab_dic)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,train_x,train_y):
        self.x_data = train_x
        self.y_data = train_y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x,y

def collate_fn(batch):
  (xx, yy) = zip(*batch)

  xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0).cuda()
  yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0).cuda()

  return xx_pad, yy_pad

# main
if __name__ == '__main__':
    with open('dic.pickle', 'rb') as f:
      vocab_dic = pickle.load(f) # 0 -> <pad> , 디코딩할 때 필요
    
    vocab_encoding = {}
    for v, k in enumerate(vocab_dic):
      vocab_encoding[k] = v # <pad> -> 0, 인코딩할 때 필요

    print('Num of vocab ', len(vocab_dic), vocab_dic[2276], vocab_encoding['<pad>'])

    train_x, train_y = prepare_sentence(ARGS.train_data, vocab_encoding)
    valid_x, valid_y = prepare_sentence(ARGS.val_data, vocab_encoding)

    print('train data\'s length :', len(train_x), '\n', train_x[0], '\n', train_y[0])
    print('valid data\'s length: ', len(valid_x), '\n', valid_x[0], '\n', valid_y[0])

    model = MY_RNN(ARGS.embedding_dim, ARGS.hidden_dim, len(vocab_dic)).to(device)

    
    if ARGS.resume == 1:
        print('resume..')
        model.load_state_dict(torch.load(ARGS.resume_model))
    print('Start training! ')

    ds = CustomDataset(train_x,train_y)
    loader_train = DataLoader(ds, batch_size = ARGS.batch_size, shuffle=True, drop_last = True, collate_fn = collate_fn)
    ds_2 = CustomDataset(valid_x,valid_y)
    loader_valid = DataLoader(ds_2, batch_size = ARGS.batch_size, shuffle=True, drop_last = True, collate_fn=collate_fn)

    LSTM_train(model, loader_train, loader_valid, vocab_dic)
    
    print('End')

#Test 없는 버전임.
