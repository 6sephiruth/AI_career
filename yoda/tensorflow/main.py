import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2



import time
import shap

from models import *
from utils import *
from data_process import *
from attribution import *
from method import *

from tqdm import trange

import pickle

import matplotlib.pyplot as plt
import matplotlib.image as img

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

HARD_MNIST_checkpoint_path = params_loaded['HARD_MNIST_checkpoint_path']
MNIST_checkpoint_path = params_loaded['MNIST_checkpoint_path']
DATASET_path = params_loaded['DATASET_path']


os.environ['TF_DETERMINISTIC_OPS'] = '0'

datadir = ['model', MNIST_checkpoint_path, 'dataset', HARD_MNIST_checkpoint_path]
mkdir(datadir)


# dataset load
if params_loaded['dataset'] == 'mnist_data':
    
    train, test = eval(params_loaded['dataset'])()
else:
    print("other dataset")

    
x_train, y_train = train
x_test, y_test = test

mnist_model = eval(params_loaded['model_train'])()



loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

if exists(f'{MNIST_checkpoint_path}/saved_model.pb'):

    mnist_model = tf.keras.models.load_model(MNIST_checkpoint_path)

else:

    # MNIST 학습 checkpoint
    checkpoint = ModelCheckpoint(MNIST_checkpoint_path, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                monitor='val_loss',
                                verbose=1)

    mnist_model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    mnist_model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint],)

    mnist_model.save(MNIST_checkpoint_path)
    mnist_model = tf.keras.models.load_model(MNIST_checkpoint_path)

mnist_model.trainable = False

ATTACK_EPS = params_loaded['attack_eps']


cw_saliency_analysis(mnist_model)











# all_data_plot(mnist_model, ATTACK_EPS)



# if exists(f'./dataset/saliency_train') and exists(f'./dataset/saliency_test'):
#     g_train = pickle.load(open(f'./dataset/saliency_train','rb'))
#     g_test = pickle.load(open(f'./dataset/saliency_test','rb'))

# else:
#     g_train, g_test = [], []

#     for i in trange(len(x_train)):
#         g_train.append(eval('vanilla_saliency')(mnist_model, x_train[i])) # (28, 28, 1)

#     for i in trange(len(x_test)):
#         g_test.append(eval('vanilla_saliency')(mnist_model, x_test[i])) # (28, 28, 1)

#     g_train, g_test = np.array(g_train), np.array(g_test)

#     pickle.dump(g_train, open(f'./dataset/saliency_train','wb'))
#     pickle.dump(g_test, open(f'./dataset/saliency_test','wb'))

# ### IG 만들기

# if exists(f'./dataset/ig_train') and exists(f'./dataset/ig_test'):
#     ig_train = pickle.load(open(f'./dataset/ig_train','rb'))
#     ig_test = pickle.load(open(f'./dataset/ig_test','rb'))

# else:
#     ig_train, ig_test = [], []

#     for i in trange(len(x_train)):
#         ig_train.append(eval('ig')(mnist_model, x_train[i])) # (28, 28, 1)

#     for i in trange(len(x_test)):
#         ig_test.append(eval('ig')(mnist_model, x_test[i])) # (28, 28, 1)

#     ig_train, ig_test = np.array(ig_train), np.array(ig_test)

#     pickle.dump(ig_train, open(f'./dataset/ig_train','wb'))
#     pickle.dump(ig_test, open(f'./dataset/ig_test','wb'))

# ### FGSM 만들기


# if exists(f'./dataset/fgsm_{ATTACK_EPS}_train') and exists(f'./dataset/fgsm_{ATTACK_EPS}_test'):
#     fgsm_train = pickle.load(open(f'./dataset/fgsm_{ATTACK_EPS}_train','rb'))
#     fgsm_test = pickle.load(open(f'./dataset/fgsm_{ATTACK_EPS}_test','rb'))

# else:
#     fgsm_train, fgsm_test = [], []

#     for i in trange(len(x_train)):
#         fgsm_train.append(eval('untargeted_fgsm')(mnist_model, x_train[i], ATTACK_EPS)) # (28, 28, 1)

#     for i in trange(len(x_test)):
#         fgsm_test.append(eval('untargeted_fgsm')(mnist_model, x_test[i], ATTACK_EPS)) # (28, 28, 1)

#     fgsm_train, fgsm_test = np.array(fgsm_train), np.array(fgsm_test)

#     pickle.dump(fgsm_train, open(f'./dataset/fgsm_{ATTACK_EPS}_train','wb'))
#     pickle.dump(fgsm_test, open(f'./dataset/fgsm_{ATTACK_EPS}_test','wb'))



# ### CW 만들기

# if exists(f'./dataset/cw_train') and exists(f'./dataset/cw_test'):
#     cw_train = pickle.load(open(f'./dataset/cw_train','rb'))
#     cw_test = pickle.load(open(f'./dataset/cw_test','rb'))

# else:
#     cw_train, cw_test = [], []

#     for i in trange(len(x_train)):
#         cw_train.append(eval('cw')(mnist_model, x_train[i])) # (28, 28, 1)

#     for i in trange(len(x_test)):
#         cw_test.append(eval('cw')(mnist_model, x_test[i])) # (28, 28, 1)

#     cw_train, cw_test = np.array(cw_train), np.array(cw_test)

#     pickle.dump(cw_train, open(f'./dataset/cw_train','wb'))
#     pickle.dump(cw_test, open(f'./dataset/cw_test','wb'))

# ### PGD 만들기

# if exists(f'./dataset/pgd_{ATTACK_EPS}_train') and exists(f'./dataset/pgd_{ATTACK_EPS}_test'):
#     pgd_train = pickle.load(open(f'./dataset/pgd_{ATTACK_EPS}_train','rb'))
#     pgd_test = pickle.load(open(f'./dataset/pgd_{ATTACK_EPS}_test','rb'))

# else:
#     pgd_train, pgd_test = [], []

#     for i in trange(len(x_train)):
#         pgd_train.append(eval('pgd')(mnist_model, x_train[i], ATTACK_EPS)) # (28, 28, 1)

#     for i in trange(len(x_test)):
#         pgd_test.append(eval('pgd')(mnist_model, x_test[i]), ATTACK_EPS) # (28, 28, 1)

#     pgd_train, pgd_test = np.array(pgd_train), np.array(pgd_test)

#     pickle.dump(pgd_train, open(f'./dataset/pgd_{ATTACK_EPS}_train','wb'))
#     pickle.dump(pgd_test, open(f'./dataset/pgd_{ATTACK_EPS}_test','wb'))

# ### MIM 만들기

# if exists(f'./dataset/mim_{ATTACK_EPS}_train') and exists(f'./dataset/mim_{ATTACK_EPS}_test'):
#     mim_train = pickle.load(open(f'./dataset/mim_{ATTACK_EPS}_train','rb'))
#     mim_test = pickle.load(open(f'./dataset/mim_{ATTACK_EPS}_test','rb'))

# else:
#     mim_train, mim_test = [], []

#     for i in trange(len(x_train)):
#         mim_train.append(eval('mim')(mnist_model, x_train[i], ATTACK_EPS)) # (28, 28, 1)

#     for i in trange(len(x_test)):
#         mim_test.append(eval('mim')(mnist_model, x_test[i]), ATTACK_EPS) # (28, 28, 1)

#     mim_train, mim_test = np.array(mim_train), np.array(mim_test)

#     pickle.dump(mim_train, open(f'./dataset/mim_{ATTACK_EPS}_train','wb'))
#     pickle.dump(mim_test, open(f'./dataset/mim_{ATTACK_EPS}_test','wb'))



### 픽셀 하이라이트 시키는 것
# highlight_pixel(mnist_model, x_train[4], 0.01)
