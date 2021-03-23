import argparse
import random
import torch
import sys
import os

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
 
parser.add_argument("--embedding_dim", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--learning_rate", type=int, default=0.0003)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--best_model", type=str, default='example')#'BiLSTM_CRF_success_best')
parser.add_argument("--best_acc", type=int, default=0)
parser.add_argument("--isneptune", type=int, default=0)
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--model_name", type=str, default='BiLSTM_CRF_success') # 저장할 모델 이름

parser.add_argument("--is_test", type=int, default=0)
parser.add_argument("--is_validation", type=int, default=500)

parser.add_argument("--valid", type=int, default=5000) # valid 이전까지는 학습만.

parser.add_argument("--train_print", type=int, default=20)
parser.add_argument("--valid_print", type=int, default=2)

# parser.add_argument("--loss_func", type=str, default='BCELoss')
parser.add_argument("--loss_func", type=str, default='BCEWithLogitsLoss')
# parser.add_argument("--loss_func", type=str, default='NLLLoss')

# parser.add_argument("--optim", type=str, default='SGD')
parser.add_argument("--optim", type=str, default='Adam')

parser.add_argument("--resume", type=int, default=0)
parser.add_argument("--resume_model", type=str, default='spacing.pt')
parser.add_argument("--train_data", type=str, default='/home/nmt3/data/en-kr/processed/aihub.valid.kr')
parser.add_argument("--val_data", type=str, default='/home/nmt3/neptune_test/datas/aihub.valid.kr')
parser.add_argument("--test_data", type=str, default='/home/nmt3/data/en-kr/processed/aihub.train.kr')

ARGS = parser.parse_args()
