import tensorflow as tf
import numpy as np
import os

import pickle

import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist

train, test = mnist.load_data()

pickle.dump(train, open(f'../mnist_train','wb'))
pickle.dump(test, open(f'../mnist_test','wb'))