import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from models import *
from utils import *
from data_process import *
from attribution import *

import matplotlib.pyplot as plt

import pickle
import time

def highlight_differnt_pixel(model, img, fgsm_eps):

    saliency_img = eval('vanilla_saliency')(model, img)

    fgsm_img = eval('untargeted_fgsm')(model, img, fgsm_eps)

    predict = model.predict(tf.expand_dims(img, 0))
    predict = np.argmax(predict)

    fgsm_saliency_img = eval('specific_vanilla_saliency')(model, fgsm_img, predict)

    extraion_arr = np.abs(saliency_img - fgsm_saliency_img)
    extraion_arr_reshpae = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshpae)

    select_small = data_sort[:30]
    select_big = data_sort[-30:]

    change_pixel = tf.expand_dims(img, 0)
    change_pixel = tf.image.grayscale_to_rgb(change_pixel)
    change_pixel = np.reshape(change_pixel, (28, 28, 3))

    small_change_pixel = change_pixel.copy()
    big_change_pixel = change_pixel.copy()

    small_change_pixel.setflags(write=1)
    big_change_pixel.setflags(write=1)

    small_change_pixel = np.reshape(small_change_pixel, (784, 3))
    big_change_pixel = np.reshape(big_change_pixel, (784, 3))

    for i in range(784):
        for j in range(30):

            if extraion_arr_reshpae[i] == select_small[j]:
                small_change_pixel[i][0] = 0.5
                small_change_pixel[i][1] = 0
                small_change_pixel[i][2] = 0


    small_change_pixel = np.reshape(small_change_pixel, (28, 28, 3))

    # plt.imshow(small_change_pixel)
    # plt.savefig("small_change_pixel.png")

    for i in range(784):
        for j in range(30):

            if extraion_arr_reshpae[i] == select_big[j]:
                big_change_pixel[i][0] = 0
                big_change_pixel[i][1] = 0.5
                big_change_pixel[i][2] = 0

    big_change_pixel = np.reshape(big_change_pixel, (28, 28, 3))

    # plt.imshow(big_change_pixel)
    # plt.savefig("big_change_pixel.png")

    return small_change_pixel, big_change_pixel

def highlight_solo_pixel(model, img):
    
    saliency_img = eval('vanilla_saliency')(model, img)

    predict = model.predict(tf.expand_dims(img, 0))
    predict = np.argmax(predict)

    extraion_arr = saliency_img
    extraion_arr_reshpae = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshpae)

    select_small = data_sort[:30]
    select_big = data_sort[-30:]

    change_pixel = tf.expand_dims(img, 0)
    change_pixel = tf.image.grayscale_to_rgb(change_pixel)
    change_pixel = np.reshape(change_pixel, (28, 28, 3))

    small_change_pixel = change_pixel.copy()
    big_change_pixel = change_pixel.copy()

    small_change_pixel.setflags(write=1)
    big_change_pixel.setflags(write=1)

    small_change_pixel = np.reshape(small_change_pixel, (784, 3))
    big_change_pixel = np.reshape(big_change_pixel, (784, 3))

    for i in range(784):
        for j in range(30):

            if extraion_arr_reshpae[i] == select_small[j]:
                small_change_pixel[i][0] = 0.5
                small_change_pixel[i][1] = 0
                small_change_pixel[i][2] = 0


    small_change_pixel = np.reshape(small_change_pixel, (28, 28, 3))

    # plt.imshow(small_change_pixel)
    # plt.savefig("solo_small_change_pixel.png")

    for i in range(784):
        for j in range(30):

            if extraion_arr_reshpae[i] == select_big[j]:
                big_change_pixel[i][0] = 0
                big_change_pixel[i][1] = 0.5
                big_change_pixel[i][2] = 0

    big_change_pixel = np.reshape(big_change_pixel, (28, 28, 3))

    # plt.imshow(big_change_pixel)
    # plt.savefig("solo_big_change_pixel.png")

    return small_change_pixel, big_change_pixel

def all_data_plot(model, eps):

    # 이걸 내가 왜 짯는지 기억이 안난다.
    # 주석좀 잘 달자.

    dataset = tf.keras.datasets.mnist
        
    (_, _), (x_test, y_test) = dataset.load_data()

    x_test = x_test.reshape((10000, 28, 28, 1))

    x_test = x_test / 255.0

    data = np.zeros([3, 10, 28, 28, 1])
    data2 = np.zeros([4, 10, 28, 28, 3])

    data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], data[0][5], data[0][6], data[0][7], data[0][8], data[0][9] = x_test[3], x_test[2], x_test[1], x_test[18], x_test[4], x_test[8], x_test[11], x_test[0], x_test[61], x_test[7]

    for i in range(10):

        perturbation_data = fgsm_perturbation(model, data[0][i], eps)
        data[1][i] = perturbation_data

        fgsm_data = untargeted_fgsm(model, data[0][i], eps)
        data[2][i] = fgsm_data

        perturbation_low_light, perturbation_high_light = highlight_solo_pixel(model, data[1][i])
        data2[0][i] = perturbation_low_light
        data2[1][i] = perturbation_high_light

        diff_low_light, diff_high_light = highlight_differnt_pixel(model, data[0][i], eps)
        data2[2][i] = diff_low_light
        data2[3][i] = diff_high_light

    fig, axs = plt.subplots(nrows=3, ncols=10, squeeze=True, figsize=(8, 8))

    for i in range(10):

        axs[0, i].imshow(data[0][i])
        axs[0,i].axis('off')

        axs[1, i].imshow(data[1][i])
        axs[1,i].axis('off')

        axs[2, i].imshow(data[2][i])
        axs[2,i].axis('off')

    fig.savefig("data.png", bbox_inches='tight', pad_inches=0.05)
    plt.close()

    fig, axs = plt.subplots(nrows=4, ncols=10, squeeze=True, figsize=(8, 8))

    for i in range(10):

        axs[0, i].imshow(data2[0][i])
        axs[0,i].axis('off')

        axs[1, i].imshow(data2[1][i])
        axs[1,i].axis('off')

        axs[2, i].imshow(data2[2][i])
        axs[2,i].axis('off')

        axs[3, i].imshow(data2[3][i])
        axs[3,i].axis('off')

    plt.savefig("data2.png", bbox_inches='tight', pad_inches=0.05)
    plt.close()

    for i in range(10):
    
        plt.imshow(data[1][i], cmap="gray")
        plt.axis('off')
        plt.savefig("data{}.png".format(i))


def cw_saliency_analysis(model):

    if exists(f'./dataset/targeted_cw_data'):
        targeted_cw = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

    else:
        dataset = tf.keras.datasets.mnist
            
        (_, _), (x_test, y_test) = dataset.load_data()

        x_test = x_test.reshape((10000, 28, 28, 1))

        x_test = x_test / 255.0

        origin_data = np.zeros([10, 28, 28, 1])
        targeted_cw = np.zeros([10, 10, 28, 28, 1])
        # data2 = np.zeros([4, 10, 28, 28, 3])

        origin_data[0], origin_data[1], origin_data[2], origin_data[3], origin_data[4], origin_data[5], origin_data[6], origin_data[7], origin_data[8], origin_data[9] = x_test[3], x_test[5], x_test[35], x_test[18], x_test[4], x_test[15], x_test[11], x_test[0], x_test[61], x_test[7]

        for i in range(10):
            for j in range(10):

                targeted_cw[i][j] = targeted_cw(model, origin_data[i], j)
                print("드디어 {}의 {} 끝났다.".format(i, j))
                
    pickle.dump(targeted_cw, open(f'./dataset/targeted_cw_data','wb'))
