import os
import smtplib
import tensorflow as tf
import numpy as np
import pickle

from email.mime.text import MIMEText

from ad_attack import *


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

    # # MIN, MAX normalization
    # x_train = (x_train - np.min(x_train))/ (np.max(x_train) - np.min(x_train))
    # x_test = (x_test - np.min(x_test))/ (np.max(x_test) - np.min(x_test))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

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

def particular_label_find(num):

    _, test = mnist_data()
    x_test, y_test = test

    particular_data = np.where(y_test == num)
    print(particular_data[0][:20])

def load_specific_dataset():

    _, test = mnist_data()
    x_test, y_test = test

    specific_dataset = np.zeros((10, 10, 28, 28, 1)) # 타겟 이미지의 saliency map
    specific_dataset[0][0], specific_dataset[0][1], specific_dataset[0][2], specific_dataset[0][3], specific_dataset[0][4], specific_dataset[0][5], specific_dataset[0][6], specific_dataset[0][7], specific_dataset[0][8], specific_dataset[0][9] = x_test[3], x_test[10], x_test[13], x_test[25], x_test[28], x_test[55], x_test[69], x_test[71], x_test[101], x_test[126]
    specific_dataset[1][0], specific_dataset[1][1], specific_dataset[1][2], specific_dataset[1][3], specific_dataset[1][4], specific_dataset[1][5], specific_dataset[1][6], specific_dataset[1][7], specific_dataset[1][8], specific_dataset[1][9] = x_test[2], x_test[5], x_test[14], x_test[29], x_test[31], x_test[37], x_test[39], x_test[40], x_test[46], x_test[57]
    specific_dataset[2][0], specific_dataset[2][1], specific_dataset[2][2], specific_dataset[2][3], specific_dataset[2][4], specific_dataset[2][5], specific_dataset[2][6], specific_dataset[2][7], specific_dataset[2][8], specific_dataset[2][9] = x_test[1], x_test[35], x_test[38], x_test[43], x_test[47], x_test[72], x_test[77], x_test[82], x_test[106], x_test[119]
    specific_dataset[3][0], specific_dataset[3][1], specific_dataset[3][2], specific_dataset[3][3], specific_dataset[3][4], specific_dataset[3][5], specific_dataset[3][6], specific_dataset[3][7], specific_dataset[3][8], specific_dataset[3][9] = x_test[18], x_test[30], x_test[32], x_test[44], x_test[51], x_test[63], x_test[68], x_test[76], x_test[87], x_test[90]
    specific_dataset[4][0], specific_dataset[4][1], specific_dataset[4][2], specific_dataset[4][3], specific_dataset[4][4], specific_dataset[4][5], specific_dataset[4][6], specific_dataset[4][7], specific_dataset[4][8], specific_dataset[4][9] = x_test[4], x_test[6], x_test[19], x_test[24], x_test[27], x_test[33], x_test[42], x_test[48], x_test[49], x_test[56]
    specific_dataset[5][0], specific_dataset[5][1], specific_dataset[5][2], specific_dataset[5][3], specific_dataset[5][4], specific_dataset[5][5], specific_dataset[5][6], specific_dataset[5][7], specific_dataset[5][8], specific_dataset[5][9] = x_test[8], x_test[15], x_test[23], x_test[45], x_test[52], x_test[53], x_test[59], x_test[102], x_test[120], x_test[127]
    specific_dataset[6][0], specific_dataset[6][1], specific_dataset[6][2], specific_dataset[6][3], specific_dataset[6][4], specific_dataset[6][5], specific_dataset[6][6], specific_dataset[6][7], specific_dataset[6][8], specific_dataset[6][9] = x_test[11], x_test[21], x_test[22], x_test[50], x_test[54], x_test[66], x_test[81], x_test[88], x_test[91], x_test[98]
    specific_dataset[7][0], specific_dataset[7][1], specific_dataset[7][2], specific_dataset[7][3], specific_dataset[7][4], specific_dataset[7][5], specific_dataset[7][6], specific_dataset[7][7], specific_dataset[7][8], specific_dataset[7][9] = x_test[0], x_test[17], x_test[26], x_test[34], x_test[36], x_test[41], x_test[60], x_test[64], x_test[70], x_test[75]
    specific_dataset[8][0], specific_dataset[8][1], specific_dataset[8][2], specific_dataset[8][3], specific_dataset[8][4], specific_dataset[8][5], specific_dataset[8][6], specific_dataset[8][7], specific_dataset[8][8], specific_dataset[8][9] = x_test[61], x_test[84], x_test[110], x_test[128], x_test[134], x_test[146], x_test[177], x_test[179], x_test[181], x_test[184]
    specific_dataset[9][0], specific_dataset[9][1], specific_dataset[9][2], specific_dataset[9][3], specific_dataset[9][4], specific_dataset[9][5], specific_dataset[9][6], specific_dataset[9][7], specific_dataset[9][8], specific_dataset[9][9] = x_test[7], x_test[9], x_test[12], x_test[16], x_test[20], x_test[58], x_test[62], x_test[73], x_test[78], x_test[92]

    return specific_dataset

def find_cw_attack(model, num):
    
    dataset = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = dataset.load_data()
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test / 255.0

    where_data = np.where(y_test == num)
    
    particular_data = x_test[where_data]

    for i in range(len(particular_data)):

        cw_data = untargeted_cw(model, particular_data[i])
        ex_cw_data = np.expand_dims(cw_data, 0)
        pred = model.predict(ex_cw_data)
        pred = np.argmax(pred)

        print("{}에서  {} 가 예상됬네".format(i, pred))