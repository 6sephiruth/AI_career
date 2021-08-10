import argparse
import os
import yaml
import time
import pickle

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import optim

import numpy as np

from helper import *
from utils import *
from models import *

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# GPU를 사용 가능할 경우 랜덤 시드를 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

#dataset_name = params_loaded['dataset_name']

directory = ['dataset', 'model']

mkdir(directory)

train = pickle.load(open(f'./mnist_train','rb'))
test = pickle.load(open(f'./mnist_test','rb'))

x_train, y_train = train
x_test, y_test = test

print(x_train.shape)


# model = eval(Mnist_CNN())
# lr = 0.1

# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#     for epoch in range(epochs):
#         model.train()
#         for xb, yb in train_dl:
#             loss_batch(model, loss_func, xb, yb, opt)

#         model.eval()
#         with torch.no_grad():
#             losses, nums = zip(
#                 *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
#             )
#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

#         print(epoch, val_loss)

# import torch.nn.functional as F

# loss_func = F.cross_entropy

# fit(epochs, model, loss_func, opt, train_dl, valid_dl)