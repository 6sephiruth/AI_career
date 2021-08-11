import tensorflow as tf
import numpy as np
import os

import pickle

import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist

train, test = mnist.load_data()

a = np.array([1., 0., 1., 1.,])
a = a.astype(bool)

b = np.array([32, 43, 22, 1])

print(b[a])
print(b[~a])