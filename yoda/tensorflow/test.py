import tensorflow as tf
import numpy as np

import pickle

import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

ad_label = pickle.load(open(f'./dataset/FGSM/0.1_label','rb'))
xai_ad_test = pickle.load(open(f'./dataset/normal_saliency_FGSM/0.1_test','rb'))

ad_label = ad_label.astype(bool)

origin_label = xai_ad_test[ad_label]
bad_label = xai_ad_test[~ad_label]

print(origin_label.shape)
print(bad_label.shape)