import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

import time
import shap

from models import *
from utils import *
from ad_attack import *
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
ATTACK_METHOD = params_loaded['attack_method']

os.environ['TF_DETERMINISTIC_OPS'] = '0'

datadir = ['model', MNIST_checkpoint_path, 'dataset', HARD_MNIST_checkpoint_path, 'dataset/'+ATTACK_METHOD]
mkdir(datadir)

ATTACK_load_path = f'dataset/{ATTACK_METHOD}'

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

# cw_saliency_analysis(mnist_model)









# i = 0

# data = np.where(y_test == i)

# for i in range(100):
#     i + 10
#     for j in range(10):
        
#         result = targeted_cw(mnist_model, x_test[data[0][i]], j)
#         result = mnist_model.predict(tf.expand_dims(result, 0))
#         result = np.argmax(result)

#         print("{}의  {} 결과는 이것입니돠 {} ".format(i, j, result))

#         if result != j:
#             break

#     print("------------------")
#     print(data[0][i])


### 픽셀 하이라이트 시키는 것
# highlight_pixel(mnist_model, x_train[4], 0.01)
