import argparse
import os
from numpy.lib.function_base import kaiser
import yaml
import time
from easydict import EasyDict
import pickle

import torch
# import torchvision.datasets as dataset

from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

from utils import *
from models import *
from ad_attack import *
from attribution import *
from method import *

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

torch.cuda.set_device(torch.device('cuda:2'))

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# GPU를 사용 가능할 경우 랜덤 시드를 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)


directory = ['dataset', 'model', 'img']

mkdir(directory)

ATTACK_TYPE = params_loaded['attack_type']
ATTACK_EPS = params_loaded['attack_eps']
XAI_TYPE = params_loaded['xai_type']

# train, test = dataset.train, dataset.test

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train_kwargs = {'batch_size': params_loaded['batch_size']}
test_kwargs = {'batch_size': params_loaded['test_batch_size']}

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset1 = datasets.MNIST('../dataset', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../dataset', train=False,
                    transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=params_loaded['learning_rate'])

scheduler = StepLR(optimizer, step_size=1, gamma= 0.07)

if exists(f'./model/mnist_cnn.pt'):
    model = torch.load(f'./model/mnist_cnn.pt')
    model.eval()

else:
    
    for epoch in range(1, params_loaded['epoch'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model, "./model/mnist_cnn.pt")

cw_saliency_analysis(model)
