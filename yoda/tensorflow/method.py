import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from models import *
from utils import *
from ad_attack import *
from attribution import *


from tqdm import trange

import pickle
import time

def highlight_differnt_saliency_pixel(origin_img, targeted_cw_img, saliency_adv_img, saliency_origin_img):

    # # 방법 1 - 그냥 빼기
    # extraion_arr = saliency_adv_img - saliency_origin_img

    # 방법 2 - absolute
    extraion_arr = np.abs(saliency_adv_img - saliency_origin_img)


    extraion_arr_reshpae = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshpae)

    select_small = data_sort[:30]
    select_big = data_sort[-30:]

    change_pixel = tf.expand_dims(origin_img, 0)
    change_pixel = tf.image.grayscale_to_rgb(change_pixel)
    change_pixel = np.reshape(change_pixel, (28, 28, 3))

    small_change_pixel = change_pixel.copy()
    big_change_pixel = change_pixel.copy()

    small_change_pixel.setflags(write=1)
    big_change_pixel.setflags(write=1)

    small_change_pixel = np.reshape(small_change_pixel, (784, 3))
    big_change_pixel = np.reshape(big_change_pixel, (784, 3))

    #########################################################

    perturbation_cw_data = targeted_cw_img - origin_img

    perturbation_background = tf.expand_dims(perturbation_cw_data, 0)
    perturbation_background = tf.image.grayscale_to_rgb(perturbation_background)
    perturbation_background = np.reshape(perturbation_background, (28, 28, 3))

    small_perturbation_pixel = change_pixel.copy()
    big_perturbation_pixel = change_pixel.copy()
    
    small_perturbation_pixel = np.reshape(small_perturbation_pixel, (784, 3))
    big_perturbation_pixel = np.reshape(big_perturbation_pixel, (784, 3))


    for i in range(784):
        for j in range(30):

            if extraion_arr_reshpae[i] == select_small[j]:

                select_small[j] = 99999  # 겹쳐지는거 방지할라고? 응급 처치 ㅋ

                small_change_pixel[i][0] = 0.5
                small_change_pixel[i][1] = 0
                small_change_pixel[i][2] = 0

                small_perturbation_pixel[i][0] = 0.5
                small_perturbation_pixel[i][1] = 0
                small_perturbation_pixel[i][2] = 0


    small_change_pixel = np.reshape(small_change_pixel, (28, 28, 3))
    small_perturbation_pixel = np.reshape(small_perturbation_pixel, (28, 28, 3))


    for i in range(784):
        for j in range(30):

            if extraion_arr_reshpae[i] == select_big[j]:

                select_big[j] = 99999  # 겹쳐지는거 방지할라고? 응급 처치 ㅋ

                big_change_pixel[i][0] = 0
                big_change_pixel[i][1] = 0.5
                big_change_pixel[i][2] = 0

                big_perturbation_pixel[i][0] = 0
                big_perturbation_pixel[i][1] = 0.5
                big_perturbation_pixel[i][2] = 0


    big_change_pixel = np.reshape(big_change_pixel, (28, 28, 3))
    big_perturbation_pixel = np.reshape(big_perturbation_pixel, (28, 28, 3))

    return small_perturbation_pixel, big_perturbation_pixel, small_change_pixel, big_change_pixel

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

        diff_low_light, diff_high_light = highlight_differnt_saliency_pixel(model, data[0][i], eps)
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

    dataset = tf.keras.datasets.mnist
        
    (_, _), (x_test, _) = dataset.load_data()

    x_test = x_test.reshape((10000, 28, 28, 1))

    x_test = x_test / 255.0


    # targeted CW dataset 만들기
    origin_data = np.zeros((10, 28, 28, 1))
    targeted_cw_data = np.zeros((10, 10, 28, 28, 1))

    origin_data[0], origin_data[1], origin_data[2], origin_data[3], origin_data[4], origin_data[5], origin_data[6], origin_data[7], origin_data[8], origin_data[9] = x_test[101], x_test[2], x_test[390], x_test[18], x_test[24], x_test[406], x_test[88], x_test[80], x_test[177], x_test[235]

    if exists(f'./dataset/targeted_cw_data'):
        targeted_cw_data = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

    else:

        for i in range(10):
            for j in range(10):

                targeted_cw_data[i][j] = targeted_cw(model, origin_data[i], j)

                ###############################################################################
                # target_result = model.predict(tf.expand_dims(targeted_cw_data[i][j], 0))
                # target_result = np.argmax(target_result)

                print("드디어 {}의 {} 끝났다.  ".format(i, j))

        pickle.dump(targeted_cw_data, open(f'./dataset/targeted_cw_data','wb'))

    perturbation_cw_data = np.zeros((10, 10, 28, 28, 1)) # perturbation 이미지

    # 새로운 주석
    saliency_origin_data = np.zeros((10, 28, 28, 1)) # 원본 이미지의 saliency map
    saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 1)) # 타겟 이미지의 saliency map

    small_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 작은 픽셀 30개 고르기
    big_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 큰 픽셀 30개 고르기

    small_perturbation_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 작은 픽셀 Overlab
    big_perturbation_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 큰 픽셀 Overlab


    for i in range(10):
        for j in range(10):

            # perturbation_cw_data[i][j] = targeted_cw_data[i][j] - origin_data[i]

            saliency_origin_data[i] = eval('vanilla_saliency')(model, origin_data[i])
            saliency_targeted_cw_data[i][j] = eval('vanilla_saliency')(model, targeted_cw_data[i][j])

            small_perturbation_targeted_cw_data[i][j], big_perturbation_targeted_cw_data[i][j], small_saliency_targeted_cw_data[i][j], big_saliency_targeted_cw_data[i][j] = highlight_differnt_saliency_pixel(origin_data[i], targeted_cw_data[i][j], saliency_targeted_cw_data[i][j], saliency_origin_data[i])

    red_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3))



    for i in range(10):
        for j in range(10):

            plt.imshow(small_saliency_targeted_cw_data[i][j])

            plt.imshow(red_saliency_targeted_cw_data[i][j], cmap="Reds")
            plt.axis('off')
            plt.savefig("data{}_{}.png".format(i, j))
