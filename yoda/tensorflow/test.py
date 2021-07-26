import tensorflow as tf
import numpy as np

import pickle

import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# print(np.where(Y_test == 0))


targeted_cw = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

