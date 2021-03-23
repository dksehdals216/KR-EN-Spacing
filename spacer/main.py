#-*- coding: utf-8 -*-
# https://github.com/jidasheng/bi-lstm-crf/blob/master/bi_lstm_crf/app/train.py
# 모델
import neptune
import os
import torch
import pickle
import torch.optim as optim
from sklearn.metrics import classification_report

from model import BiRnnCrf
from config import ARGS
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor
from torch.utils.data import DataLoader

from preprocess import encode_string, get_label, prepare_sentence
from postprocess import int_to_string, CustomDataset, collate_fn

torch.manual_seed(1)

#gpu number setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:6" if use_cuda else "cpu")
torch.cuda.set_device(device)

if use_cuda == 1:
  print('cuda is available')

best_acc = 0

def cal_acc(outputs, y, mask):
    tot_acc = 0

    for i, (Os, Ys, Ms) in enumerate(zip(outputs, y.to(device), mask)):

      Ms = Ms.gt(0)

      Os = torch.LongTensor(Os)
      Ys = Ys.masked_select(Ms)
      sent_leng = len(Ys)

      num_corrects = (Os.to(device=device, dtype=torch.int64) == Ys.to(device=device, dtype=torch.int64)).sum().item()

      tot_acc += num_corrects/sent_leng

    return tot_acc/ARGS.batch_size

def eval_model(model, device, dataloader, desc, vocab, epoch):
    tot_val_loss, f1_total, tot_acc = 0, 0, 0
    global best_acc
    best_acc = 0
    state = 'VALIDATION'

    if ARGS.is_test == 1:
      model.load_state_dict(torch.load('{}.pt'.format(ARGS.best_model)))
      state = 'TEST'
      model.to(device)

    model.eval()
    for i, (sentence, y) in enumerate(dataloader):

        with torch.no_grad():
            # eval
            loss = model.loss(sentence, y)

            mask = sentence.ne(0)
            loss = loss.masked_select(mask).mean()
            tot_val_loss += loss

            # ACC caculate
            _, predicted_Y = model(sentence)
            tot_acc += cal_acc(predicted_Y, y, mask)

            # F1 score 
            y_f1 = y[0].masked_select(mask[0])
            py_f1 = predicted_Y[0]

            report = classification_report(y_f1.cpu(), py_f1, target_names=['non_spacing', 'spacing'], output_dict=True)
            f1s = report['spacing']['f1-score']
            f1_total += f1s

            if ARGS.is_test == 1:
                with open("/home/nmt3/neptune_test/nmt_pytorch/results/trans/originalvalid.kr.spacing", "a") as f:
                    predicted_Ys = int_to_string(sentence, predicted_Y, vocab)
                    
                    for p_y in predicted_Ys:
                        f.write(p_y + '\n')
                
            if (i+1) % (ARGS.valid_print) == 0:
                predicted_Ys = int_to_string(sentence, predicted_Y, vocab)
                print('=>', predicted_Ys[0])

                avg_loss = (tot_val_loss/ARGS.valid_print)/ARGS.batch_size
                avg_acc = tot_acc/ARGS.valid_print
                avg_f1 = f1_total/ARGS.valid_print

                print('[Epoch {i}] [{s} LOSS] {l:.4f} [{s} ACC] {a:.4f} [{s} f1] {f:.4f}\n'.format(i=epoch, s=state, l=avg_loss, a=avg_acc, f=avg_f1))
                tot_val_loss, f1_total, tot_acc = 0, 0, 0

                if ARGS.isneptune == 1 and ARGS.is_test == 0:
                  neptune.log_metric('{} loss'.format(state), avg_loss)
                  neptune.log_metric('{} acc'.format(state), avg_acc)
                  neptune.log_metric('{} f1'.format(state), avg_f1)
                
                if avg_acc > best_acc and  ARGS.is_test != 1:
                  best_acc = avg_acc
                  print('[Best ACC] {a:.4f}\n\n'.format(a=best_acc))

                  # 모델 저장
                  cur_weight = model.state_dict()
                  torch.save(cur_weight, '{}.pt'.format(ARGS.best_model))
                  
    model.train()

    if ARGS.is_test == 1:
        f.close()

def train(model, loader_train, loader_valid, vocab_dic):
    global best_acc
    num_corrects, num_total, count, epoch = 0, 0, 0, 0

    loss_total = 0
    acc_total, f1_total = 0, 0

    if ARGS.isneptune == 1:
        neptune.init(project_qualified_name='', # neptune ID
              api_token='', # neptune API token
            )
        neptune.create_experiment(name=ARGS.model_name)

    if ARGS.optim == "Adam":
      optimizer = optim.Adam(model.parameters(), lr=ARGS.learning_rate)
    elif ARGS.optim == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=ARGS.learning_rate)

    model.to(device)

    while True: 
        epoch += 1
        model.train()

        for i, (sentence, y) in enumerate(loader_train):
            model.zero_grad()

            loss = model.loss(sentence.to(device), y.to(device))
            loss.backward()
            optimizer.step()

            loss_total += loss

            if (i+1) % ARGS.train_print == 0:
              
              train_loss_avg = loss_total/(ARGS.train_print)
              print('[Epoch {i}, {e}] [LOSS] {l}'.format(i=epoch, e=(i+1), l=train_loss_avg/(ARGS.batch_size)))

              if ARGS.isneptune == 1:
                  neptune.log_metric('train_loss', train_loss_avg/(ARGS.batch_size))
              
              loss_total = 0

        eval_model(model, "cuda", dataloader=loader_valid, desc="eval", vocab=vocab_dic, epoch=epoch)

# main
if __name__ == '__main__':
    tag_to_ix = {"I": 0, "B": 1}
    with open('aihub_dic.pickle', 'rb') as f:
      vocab_dic = pickle.load(f) # 0 -> <pad> , 디코딩할 때 필요
    
    vocab_encoding = {}
    for v, k in enumerate(vocab_dic):
      vocab_encoding[k] = v # <pad> -> 0, 인코딩할 때 필요
    
    model = BiRnnCrf(len(vocab_dic), len(tag_to_ix), ARGS.embedding_dim, ARGS.hidden_dim)#.to(device)

    # for test 
    if ARGS.is_test == 1:

      test_x, test_y = prepare_sentence(ARGS.test_data, vocab_encoding)
      ds = CustomDataset(test_x, test_y)
      loader_test = DataLoader(ds, batch_size = ARGS.batch_size, shuffle=False, drop_last = False, collate_fn = collate_fn)
      print('Start Testing! ')
      eval_model(model, device, dataloader=loader_test, desc="eval", vocab=vocab_dic, epoch=1)

    # for train and validate
    else:
      print('Num of vocab ', len(vocab_dic), vocab_dic[2276], vocab_encoding['<pad>'])

      train_x, train_y = prepare_sentence(ARGS.train_data, vocab_encoding)
      valid_x, valid_y = prepare_sentence(ARGS.val_data, vocab_encoding)

      print('train data\'s length :', len(train_x))
      print('valid data\'s length: ', len(valid_x))
      
      if ARGS.resume == 1:
          print('resume..')
          model.load_state_dict(torch.load(ARGS.resume_model))
      print('Start training!! ')

      ds = CustomDataset(train_x, train_y)
      loader_train = DataLoader(ds, batch_size = ARGS.batch_size, shuffle=True, drop_last = True, collate_fn = collate_fn)
      ds_2 = CustomDataset(valid_x, valid_y)
      loader_valid = DataLoader(ds_2, batch_size = ARGS.batch_size, shuffle=False, drop_last = True, collate_fn=collate_fn)

      train(model, loader_train, loader_valid, vocab_dic)
      print('End')
