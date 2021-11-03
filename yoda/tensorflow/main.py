import argparse
import os
from tensorflow.python.framework import dtypes
import yaml
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

import time

from models import *
from utils import *
from ad_attack import *
from attribution import *
from method import *

from tqdm import trange

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')

for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

os.environ['TF_DETERMINISTIC_OPS'] = '0'


ATTACK_METHOD = params_loaded['attack_method']
DATASET = params_loaded['dataset']

datadir = ['model', 'model/' + DATASET, 'dataset', 'dataset/' + ATTACK_METHOD, 'img']
mkdir(datadir)

ATTACK_EPS = params_loaded['attack_eps']

# dataset load
if DATASET == 'mnist':
    train, test = mnist_data()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

elif DATASET == 'cifar10':
    train, test = cifar10_data()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
x_train, y_train = train
x_test, y_test = test

model = eval(params_loaded['model_train'])()

# checkpoint_path = f'model/{DATASET}'

# if exists(f'model/{DATASET}/saved_model.pb'):

#     model = tf.keras.models.load_model(checkpoint_path)

# else:

#     # MNIST 학습 checkpoint
#     checkpoint = ModelCheckpoint(checkpoint_path, 
#                                 save_best_only=True, 
#                                 save_weights_only=True, 
#                                 monitor='val_loss',
#                                 verbose=1)
#     if DATASET == 'mnist':

#         model.compile(optimizer='adam',
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy'])

#         model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint],)
    
#     # if DATASET == 'cifar10':

#     #     model.compile(optimizer='adam',
#     #                 loss='sparse_categorical_crossentropy',
#     #                 metrics=['accuracy'])

#     #     model.fit(x_train, y_train, epochs=40, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint],)

#     # model.save(checkpoint_path)
#     # model = tf.keras.models.load_model(checkpoint_path)

# # model.trainable = False

# model = mnist_model()
# model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test))

# print(model.evaluate(x_test, y_test))

# model.trainable = False
# model.save('./model/kkkk')
model = tf.keras.models.load_model('./model/kkkk')

# print(model.summary())
# print(model.evaluate(x_test, y_test))

model.trainable = False
# modify_cw_data(model)
# find_target_cw_attack(model, 2)

# make_specific_cw(model,0, 35)


# cw_data = pickle.load(open(f'./dataset/cw_specific/0','rb'))

# for i in range(10):

#     pred = model.predict(tf.expand_dims(cw_data[i], 0))
#     pred = np.argmax(pred)
#     print(pred)


# modify_cw_data()

change_cw_data = pickle.load(open(f'./dataset/cw_specific/change_data','rb'))

# comparision_neuron_activation(model, change_cw_data)

# for i in range(10):
#     for j in range(10):
        
#         plt.imshow(change_cw_data[i][j])
#         plt.savefig("./img/k_{}_{}".format(i,j))


for i in range(10):
    for j in range(10):

        pred = model.predict(tf.expand_dims(change_cw_data[i][j], 0))
        pred = np.argmax(pred)
        print(pred)
        
    print("--------")




#print(np.argmax(model.predict(cw_data)))




# for i in range(10):
#     for j in range(10):
        
#         targeted_cw_data = pickle.load(open(f'./dataset/cw_specific/{i}','rb'))

#         data = targeted_cw_data[j]
#         pred = model.predict(tf.expand_dims(data, 0))
#         pred = np.argmax(pred)

#         print(pred)





# ff = np.where(y_test == 0)
# comparision_neuron_activation(model, x_test[ff][:10])

# print(len(model.layers))
# print(model.summary())
# find_target_cw_attack(model, 9)


#print(intermediate_output)


# x1, y1, y2 = [0, 896], [25, 25], [50, 50]
# plt.plot(x1, y1,'w', x1, y2, 'w')
# # plt.plot(x1, y2, 'b')

# plt.imshow(k)
# plt.savefig('./kk.png')

# print(k.shape)

#print(np.array(model.layers[0].output.shape))
# print(len(model.layers))
#print(len(model.layers[0].output.shape))






# model = tf.keras.models.load_model('model/summary_mnist')

# targeted_cw_data = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

# cw_specific_dataset = pickle.load(open(f'./dataset/cw_specific_dataset','rb'))







# for i in range(10):
#     input_data = cw_specific_dataset[i]

#     hideen_0 = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
#     hideen_1 = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)
#     hideen_2 = tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
#     hideen_3 = tf.keras.Model(inputs=model.input, outputs=model.layers[3].output)
#     hideen_4 = tf.keras.Model(inputs=model.input, outputs=model.layers[4].output)
#     hideen_5 = tf.keras.Model(inputs=model.input, outputs=model.layers[5].output)
#     hideen_6 = tf.keras.Model(inputs=model.input, outputs=model.layers[6].output)

#     output_0 = np.reshape(hideen_0(input_data).numpy(), -1)
#     output_1 = np.reshape(hideen_1(input_data).numpy(), -1)
#     output_2 = np.reshape(hideen_2(input_data).numpy(), -1)
#     output_3 = np.reshape(hideen_3(input_data).numpy(), -1)
#     output_4 = np.reshape(hideen_4(input_data).numpy(), -1)
#     output_5 = np.reshape(hideen_5(input_data).numpy(), -1)
#     output_6 = np.reshape(hideen_6(input_data).numpy(), -1)

#     all_data = np.concatenate([output_0, output_1, output_2, output_3, output_4, output_5, output_6])

#     analysis_data = np.zeros(4)

#     for i in range(len(all_data)):
        
#         if all_data[i] ==0 :
#             analysis_data[0] += 1
#         elif 4> all_data[i] > 0:
#             analysis_data[1] += 1
#         elif 7> all_data[i] >= 4:
#             analysis_data[2] += 1
#         elif all_data[i] >= 7:
#             analysis_data[3] += 1

#     # print(all_data.shape)
#     print("min {}  max {:3f}  mean{:3f}".format(np.min(all_data), np.max(all_data), np.mean(all_data)))

#     for i in range(4):
#         print(analysis_data[i])
#     print("-----------------")