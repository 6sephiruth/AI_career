import tensorflow as tf
import numpy as np

import pickle

import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

a = [10]
b = [1]

print(np.greater(a, b))







# 정상 데이터와 오토인코더로 복원된 데이터 오차 값 계산
normal_data = tf.reshape(normal_data, (len(normal_data), 784))
train_recon = autoencoder.predict(normal_data)
recon_normal_data = tf.reshape(train_recon, (len(train_recon), 784))
train_loss = tf.keras.losses.mae(recon_normal_data, normal_data)

# :threshold: train loss 값의 평균과 표준편차 값
threshold = np.mean(train_loss) + np.std(train_loss)

print("Threshold: ", threshold)

# (정상 데이터 or 적대적 예제) 분류하고자 하는 데이터의 오차 값 계산
classification_data = tf.reshape(classification_data, (len(classification_data), 784))
recon_classification_data = autoencoder.predict(classification_data)
recon_classification_data = tf.reshape(recon_classification_data,(len(test_recon), 784))

test_loss = tf.keras.losses.mae(recon_classification_data, classification_data)

# 정상 데이터의 복원 오차 값 보다 클 경우, 적대적 예제로 판단
preds = np.greater(test_loss,threshold)

# (정상 데이터 or 적대적 예제) 분류 인공지능 모델 성능 평가
print('Accuracy = %f' % accuracy_score(classification_label), preds))
print('Precision = %f' % precision_score(classification_label, preds))
print('Recall = %f\n' % recall_score(classification_label, preds))

fpr, tpr, threshold = metrics.roc_curve(classification_label, test_loss)
auc = metrics.auc(fpr, tpr)
print('AUC = %f' % auc)