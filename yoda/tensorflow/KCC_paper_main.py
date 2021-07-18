import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

from models import *
from utils import *
from data_process import *
from attribution import *
from method import *

from tqdm import trange

import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.metrics as metrics

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

### Saliency map 만들기
if exists(f'./dataset/saliency_train') and exists(f'./dataset/saliency_test'):
    g_train = pickle.load(open(f'./dataset/saliency_train','rb'))
    g_test = pickle.load(open(f'./dataset/saliency_test','rb'))

else:
    g_train, g_test = [], []

    for i in trange(len(x_train)):
        g_train.append(eval('vanilla_saliency')(mnist_model, x_train[i])) # (28, 28, 1)

    for i in trange(len(x_test)):
        g_test.append(eval('vanilla_saliency')(mnist_model, x_test[i])) # (28, 28, 1)

    g_train, g_test = np.array(g_train), np.array(g_test)

    pickle.dump(g_train, open(f'./dataset/saliency_train','wb'))
    pickle.dump(g_test, open(f'./dataset/saliency_test','wb'))

# adversarial dataset을 저장하기 위한 변수 선언
x_adversarial_test = np.zeros_like(x_test)

# 테스트 데이터 셋 중 랜덤하게 절반을 adversarial example을 만들고자 함. * True: 정상 데이터,  False 적대적 데이터
random_data = tf.random.uniform(
    shape=[len(x_test)], minval=0, maxval=1, dtype=tf.dtypes.float32)

# 0~1까지를 랜덤하게 한 뒤, 0.5 가 넘으면 정상 데이터, 낮으면 적대적 데이터
adversarial_list = tf.cast(random_data>=0.0, tf.bool).numpy()

### FGSM 만들기
ATTACK_EPS = params_loaded['attack_eps']

if exists(f'./dataset/fgsm_{ATTACK_EPS}_train') and exists(f'./dataset/fgsm_{ATTACK_EPS}_test') and exists(f'./dataset/x_adversarial_{ATTACK_EPS}_test') and exists(f'./dataset/adversarial_{ATTACK_EPS}_list'):
    fgsm_train = pickle.load(open(f'./dataset/fgsm_{ATTACK_EPS}_train','rb'))
    fgsm_test = pickle.load(open(f'./dataset/fgsm_{ATTACK_EPS}_test','rb'))
    x_adversarial_test = pickle.load(open(f'./dataset/x_adversarial_{ATTACK_EPS}_test','rb'))
    adversarial_list = pickle.load(open(f'./dataset/adversarial_{ATTACK_EPS}_list','rb'))

else:
    fgsm_train, fgsm_test = [], []

    for i in trange(len(x_train)):
        fgsm_train.append(eval('untargeted_fgsm')(mnist_model, x_train[i], ATTACK_EPS)) # (28, 28, 1)

    for i in trange(len(x_test)):
        fgsm_test.append(eval('untargeted_fgsm')(mnist_model, x_test[i], ATTACK_EPS)) # (28, 28, 1)

        adv_x = eval('untargeted_fgsm')(mnist_model, x_test[i], ATTACK_EPS)

        if y_test[i] == np.argmax(mnist_model(tf.expand_dims(adv_x, 0))):
            x_adversarial_test[i] = x_test[i]
            adversarial_list[i] = False
        else:
            x_adversarial_test[i] = adv_x[0]
            
    fgsm_train, fgsm_test = np.array(fgsm_train), np.array(fgsm_test)

    pickle.dump(fgsm_train, open(f'./dataset/fgsm_{ATTACK_EPS}_train','wb'))
    pickle.dump(fgsm_test, open(f'./dataset/fgsm_{ATTACK_EPS}_test','wb'))
    pickle.dump(x_adversarial_test, open(f'./dataset/x_adversarial_{ATTACK_EPS}_test','wb'))
    pickle.dump(adversarial_list, open(f'./dataset/adversarial_{ATTACK_EPS}_list','wb'))
    
### Adversarial Saliency map 만들기
if exists(f'./dataset/fgsm_{ATTACK_EPS}_saliency_test'):
    fgsm_g_test = pickle.load(open(f'./dataset/fgsm_{ATTACK_EPS}_saliency_test','rb'))

else:
    fgsm_g_test = []

    for i in trange(len(x_adversarial_test)):
        fgsm_g_test.append(eval('vanilla_saliency')(mnist_model, x_adversarial_test[i])) # (28, 28, 1)

    fgsm_g_test = np.array(fgsm_g_test)

    pickle.dump(fgsm_g_test, open(f'./dataset/fgsm_{ATTACK_EPS}_saliency_test','wb'))



x_saliency_train = g_train
x_adversarial_saliency_test = x_adversarial_test
adversarial_list = adversarial_list

# 정상 데이터와, 적대적 데이터 분류
cut = int(len(adversarial_list)/2)

normal_saliency_data = x_adversarial_saliency_test[:cut]
anomalous_saliency_data = x_adversarial_saliency_test[cut:]

autoencoder = eval('AnomalyDetector')()

autoencoder.compile(optimizer='adam', loss='mae')

checkpoint_path = 'autoencoder_model/ckpt/autoencoder.ckpt'

checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='val_loss',
                             verbose=1)

history = autoencoder.fit(x_saliency_train, x_saliency_train, 
          epochs=5,
          batch_size=32,
          validation_data=(normal_saliency_data, normal_saliency_data),
          shuffle=True,
          callbacks=[checkpoint])

autoencoder.load_weights(checkpoint_path)
autoencoder.trainable = False

# train loss

#x_saliency_train, x_adversarial_saliency_test, adversarial_list= load_saliency_dataset()

x_saliency_train_reshape = tf.reshape(x_saliency_train,(len(x_saliency_train),784))
train_reconstructions = autoencoder.predict(x_saliency_train)
train_reconstructions_reshape = tf.reshape(train_reconstructions,(len(train_reconstructions), 784))

train_loss = tf.keras.losses.mae(train_reconstructions_reshape, x_saliency_train_reshape)

# threshold -> train loss 값의 평균과 표준편차 값

threshold = np.mean(train_loss) + np.std(train_loss)
#threshold = np.mean(train_loss)

print("Threshold: ", threshold)

# test
#x_saliency_train, x_adversarial_saliency_test, adversarial_list= load_saliency_dataset()
x_saliency_test_reshape = tf.reshape(anomalous_saliency_data,(len(anomalous_saliency_data),784))
test_reconstructions = autoencoder.predict(anomalous_saliency_data)
test_reconstructions_reshape = tf.reshape(test_reconstructions,(len(test_reconstructions), 784))

test_loss = tf.keras.losses.mae(test_reconstructions_reshape, x_saliency_test_reshape)

x_adversarial_saliency_test_reshape = tf.reshape(x_adversarial_saliency_test,(len(x_adversarial_saliency_test),784))
test_reconstructions = autoencoder.predict(x_adversarial_saliency_test)
test_reconstructions_reshape = tf.reshape(test_reconstructions,(len(test_reconstructions), 784))

test_loss = tf.keras.losses.mae(test_reconstructions_reshape, x_adversarial_saliency_test_reshape)

preds = np.greater(test_loss,threshold)

print('Accuracy = %f' % accuracy_score(adversarial_list, preds))
print('Precision = %f' % precision_score(adversarial_list, preds))
print('Recall = %f\n' % recall_score(adversarial_list, preds))

fpr, tpr, threshold = metrics.roc_curve(adversarial_list, test_loss)
auc = metrics.auc(fpr, tpr)
print('AUC = %f' % auc)
