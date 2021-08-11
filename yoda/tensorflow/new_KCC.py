import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.metrics as metrics

import time

from models import *
from utils import *
from ad_attack import *
from attribution import *
from method import *

from tqdm import trange

import pickle

# seed = 0
# tf.random.set_seed(seed)
# np.random.seed(seed)

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

os.environ['TF_DETERMINISTIC_OPS'] = '0'

ATTACK_METHOD = params_loaded['attack_method']
DATASET = params_loaded['dataset']
XAI_METHOD = params_loaded['xai_method']
ATTACK_EPS = params_loaded['attack_eps']

datadir = ['model', 'model/' + DATASET, 'dataset', 'dataset/' + ATTACK_METHOD, 'img']
mkdir(datadir)

# dataset load
if DATASET == 'mnist':
    train, test = mnist_data()
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

elif DATASET == 'cifar10':
    train, test = cifar10_data()
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
x_train, y_train = train
x_test, y_test = test

model = eval(params_loaded['model_train'])()

cnn_checkpoint_path = f'model/{DATASET}'

if exists(f'model/{DATASET}/saved_model.pb'):

    model = tf.keras.models.load_model(cnn_checkpoint_path)

else:

    # MNIST 학습 checkpoint
    cnn_checkpoint = ModelCheckpoint(cnn_checkpoint_path, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                monitor='val_loss',
                                verbose=1)
    if DATASET == 'mnist':

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test), callbacks=[cnn_checkpoint],)
    
    elif DATASET == 'cifar10':

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=40, shuffle=True, validation_data=(x_test, y_test), callbacks=[cnn_checkpoint],)

    model.save(cnn_checkpoint_path)
    model = tf.keras.models.load_model(cnn_checkpoint_path)

model.trainable = False

g_train = pickle.load(open(f'./dataset/{XAI_METHOD}/train','rb'))

# ad_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))
# ad_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','rb'))
# xai_ad_test = pickle.load(open(f'./dataset/{XAI_METHOD}_{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))


ad_test = pickle.load(open(f'./dataset/FGSM/{ATTACK_EPS}_test','rb'))
ad_label = pickle.load(open(f'./dataset/FGSM/{ATTACK_EPS}_label','rb'))
xai_ad_test = pickle.load(open(f'./dataset/normal_saliency_FGSM/{ATTACK_EPS}_test','rb'))


ad_label = ad_label.astype(bool)

xai_origin = xai_ad_test[~ad_label]
xai_ad = xai_ad_test[ad_label]

if len(xai_origin) > len(xai_ad):
    xai_origin = xai_origin[:len(xai_ad)]

else:
    xai_ad = xai_ad[:len(xai_origin)]

ad_test = np.concatenate([xai_origin, xai_ad], 0)

ad_label = np.ones(len(ad_test))
ad_label[:len(xai_origin)] = 0

# ad_label = ad_label.astype(bool)

auto_checkpoint_path = f'model/autoencoder'

if exists(f'model/autoencoder/saved_model.pb'):
    
    autoencoder = tf.keras.models.load_model(auto_checkpoint_path)

else:

    autoencoder = eval('AnomalyDetector')()

    autoencoder.compile(optimizer='adam', loss='mae')


    auto_checkpoint = ModelCheckpoint(auto_checkpoint_path, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    monitor='val_loss',
                                    verbose=1)

    history = autoencoder.fit(g_train, g_train, 
            epochs=5,
            batch_size=32,
            validation_data=(xai_origin, xai_origin),
            shuffle=True,
            callbacks=[auto_checkpoint])

    autoencoder.save(auto_checkpoint_path)
    autoencoder = tf.keras.models.load_model(auto_checkpoint_path)

autoencoder.trainable = False

g_train_reshape = tf.reshape(g_train, (len(g_train), 784))
train_recon = autoencoder.predict(g_train)
train_recon_reshape = tf.reshape(train_recon, (len(train_recon), 784))

train_loss = tf.keras.losses.mae(train_recon_reshape, g_train_reshape)

# threshold -> train loss 값의 평균과 표준편차 값

threshold = np.mean(train_loss) + np.std(train_loss)

print("Threshold: ", threshold)

# test
x_saliency_test_reshape = tf.reshape(ad_test, (len(ad_test), 784))
test_recon = autoencoder.predict(ad_test)
test_reconstructions_reshape = tf.reshape(test_recon,(len(test_recon), 784))

test_loss = tf.keras.losses.mae(test_reconstructions_reshape, x_saliency_test_reshape)

preds = np.greater(test_loss, threshold)


print('Accuracy = %f' % accuracy_score(ad_label, preds))
print('Precision = %f' % precision_score(ad_label, preds))
print('Recall = %f\n' % recall_score(ad_label, preds))

fpr, tpr, threshold = metrics.roc_curve(ad_label, test_loss)
auc = metrics.auc(fpr, tpr)
print('AUC = %f' % auc)