import tensorflow as tf
import numpy as np

from data_process import *



mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

zero = np.where(Y_test == 0)

k = np.zeros([10, 3, 3])
print(k.shape)