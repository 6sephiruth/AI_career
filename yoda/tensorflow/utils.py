import os
import smtplib
import tensorflow as tf
import numpy as np

from email.mime.text import MIMEText


def mnist_data():
    
    dataset = tf.keras.datasets.mnist
        
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def cifar10_data():

    dataset = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    
    x_train = x_train.reshape((50000, 32, 32, 3))
    x_test = x_test.reshape((10000, 32, 32, 3))

    # MIN, MAX normalization
    x_train = (x_train - np.min(x_train))/ (np.max(x_train) - np.min(x_train))
    x_test = (x_test - np.min(x_test))/ (np.max(x_test) - np.min(x_test))
    
    return (x_train, y_train), (x_test, y_test)



def exists(pathname):
    return os.path.exists(pathname)

def mkdir(dir_names):
    for d in dir_names:
        if not os.path.exists(d):
            os.mkdir(d)

def email_send():
    # 아직 코드 미 작성
    # 보안 코드 작성 뭐시기 해야함 ㅜ
    # https://yeolco.tistory.com/93
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('6sephiruth@gmail.com')
