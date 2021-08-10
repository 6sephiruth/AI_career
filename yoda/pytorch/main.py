import argparse
import os
import yaml
import time
from easydict import EasyDict

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms

import numpy as np

from helper import *
from utils import *
from models import *

import saliency.core as saliency

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

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

# designate gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '2' # params_loaded['gpu_num']

directory = ['dataset', 'model']

mkdir(directory)


train, test = dataset.train, dataset.test

print(train)

# train_data_loader = torch.utils.data.DataLoader(dataset=train,
#                                                batch_size=params_loaded['batch_size'],
#                                                shuffle=True,
#                                                drop_last=True)

# test_data_loader = torch.utils.data.DataLoader(dataset=test,
#                                                batch_size=params_loaded['batch_size'],
#                                                shuffle=False,
#                                                drop_last=True)


# model_name = params_loaded['model_name']

# model = eval(model_name)().to(device)

# criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# total_batch = len(train_data_loader)
# print('총 배치의 수 : {}'.format(total_batch))

# if exists(f'model/{dataset_name}.pt'):
    
#     model.load_state_dict(torch.load(f'model/{dataset_name}.pt'))
#     model.eval()
# else:
#     for epoch in range(params_loaded['epoch_count']):
#         avg_cost = 0

#         for X, Y in train_data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
#             # image is already size of (28x28), no reshape
#             # label is not one-hot encoded
#             X = X.to(device)
#             Y = Y.to(device)

#             optimizer.zero_grad()
#             hypothesis = model(X)
#             cost = criterion(hypothesis, Y)
#             cost.backward()
#             optimizer.step()

#             avg_cost += cost / total_batch

#         print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

#         torch.save(model.state_dict(), f'model/{dataset_name}.pt')

# # # 학습을 진행하지 않을 것이므로 torch.no_grad()
# # with torch.no_grad():
# #     X_test = test.test_data.view(len(test), 1, 28, 28).float().to(device)
# #     Y_test = test.test_labels.to(device)

# #     prediction = model(X_test)
# #     correct_prediction = torch.argmax(prediction, 1) == Y_test
# #     accuracy = correct_prediction.float().mean()
# #     print('Accuracy:', accuracy.item())

# report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)



# for x, y in test_data_loader:
#     x, y = x.to(device), y.to(device)
    
#     x_fgm = fast_gradient_method(model, x, 0.5, np.inf)

#     _, y_pred = model(x).max(1)

#     _, y_pred_fgm = model(x_fgm).max(1)

#     # report.nb_test += y.size(0)
#     # report.correct += y_pred.eq(y).sum().item()
#     # report.correct_fgm += y_pred_fgm.eq(y).sum().item()

#     # print(
#     #     "test acc on clean examples (%): {:.3f}".format(
#     #         report.correct / report.nb_test * 100.0
#     #     )
#     # )
#     # print(
#     #     "test acc on FGM adversarial examples (%): {:.3f}".format(
#     #         report.correct_fgm / report.nb_test * 100.0
#     #     )
#     # )
#     # time.sleep(3)

