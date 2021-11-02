import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from models import *
from utils import *
from ad_attack import *
from attribution import *


from tqdm import trange

import pickle
import time

def highlight_differnt_saliency_pixel(origin_img, targeted_cw_img, saliency_adv_img, saliency_origin_img):
    
    # 방법 1 - 그냥 빼기
    extraion_arr = saliency_adv_img - saliency_origin_img


    extraion_arr_reshape = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshape)

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

    perturbation_cw_data = (targeted_cw_img - origin_img)

    # perturbation_cw_data = np.reshape(perturbation_cw_data, (-1))

    # for i in range(len(perturbation_cw_data)):

    #     if perturbation_cw_data[i] != 0:
    #         perturbation_cw_data[i] += 0.1
    # perturbation_cw_data = np.reshape(perturbation_cw_data, (28, 28, 1))    

    perturbation_background = tf.expand_dims(perturbation_cw_data, 0)
    perturbation_background = tf.image.grayscale_to_rgb(perturbation_background)
    perturbation_background = np.reshape(perturbation_background, (28, 28, 3))


    small_perturbation_pixel = perturbation_background.copy()
    big_perturbation_pixel = perturbation_background.copy()
    
    small_perturbation_pixel.setflags(write=1)
    big_perturbation_pixel.setflags(write=1)

    small_perturbation_pixel = np.reshape(small_perturbation_pixel, (784, 3))
    big_perturbation_pixel = np.reshape(big_perturbation_pixel, (784, 3))

    for i in range(30):
        for j in range(784):

            if extraion_arr_reshape[j] == select_small[i]:
                
                if small_change_pixel[j][0] == 0.5 and small_perturbation_pixel[j][0] == 0.5:        
                    continue

                small_change_pixel[j][0] = 0.5
                small_change_pixel[j][1] = 0
                small_change_pixel[j][2] = 0

                small_perturbation_pixel[j][0] = 0.5
                small_perturbation_pixel[j][1] = 0
                small_perturbation_pixel[j][2] = 0

                break


    small_change_pixel = np.reshape(small_change_pixel, (28, 28, 3))
    small_perturbation_pixel = np.reshape(small_perturbation_pixel, (28, 28, 3))

    for i in range(30):
        for j in range(784):

            if extraion_arr_reshape[j] == select_big[i]:

                if big_change_pixel[j][1] == 0.5 and big_perturbation_pixel[j][1] == 0.5:
                    continue

                big_change_pixel[j][0] = 0
                big_change_pixel[j][1] = 0.5
                big_change_pixel[j][2] = 0

                big_perturbation_pixel[j][0] = 0
                big_perturbation_pixel[j][1] = 0.5
                big_perturbation_pixel[j][2] = 0

                break

    big_change_pixel = np.reshape(big_change_pixel, (28, 28, 3))
    big_perturbation_pixel = np.reshape(big_perturbation_pixel, (28, 28, 3))

    return perturbation_background, small_perturbation_pixel, big_perturbation_pixel, small_change_pixel, big_change_pixel


def abs_highlight_differnt_saliency_pixel(origin_img, targeted_cw_img, saliency_adv_img, saliency_origin_img):

    # # 방법 1 - 그냥 빼기
    # extraion_arr = saliency_adv_img - saliency_origin_img

    extraion_arr = np.abs(saliency_adv_img - saliency_origin_img)


    extraion_arr_reshape = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshape)

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

    perturbation_cw_data = (targeted_cw_img - origin_img)

    # perturbation_cw_data = np.reshape(perturbation_cw_data, (-1))

    # for i in range(len(perturbation_cw_data)):

    #     if perturbation_cw_data[i] != 0:
    #         perturbation_cw_data[i] += 0.1
    # perturbation_cw_data = np.reshape(perturbation_cw_data, (28, 28, 1))


    perturbation_background = tf.expand_dims(perturbation_cw_data, 0)
    perturbation_background = tf.image.grayscale_to_rgb(perturbation_background)
    perturbation_background = np.reshape(perturbation_background, (28, 28, 3))


    small_perturbation_pixel = perturbation_background.copy()
    big_perturbation_pixel = perturbation_background.copy()
    
    small_perturbation_pixel.setflags(write=1)
    big_perturbation_pixel.setflags(write=1)

    small_perturbation_pixel = np.reshape(small_perturbation_pixel, (784, 3))
    big_perturbation_pixel = np.reshape(big_perturbation_pixel, (784, 3))

    for i in range(30):
        for j in range(784):

            if extraion_arr_reshape[j] == select_small[i]:
                
                if small_change_pixel[j][0] == 0.5 and small_perturbation_pixel[j][0] == 0.5:        
                    continue

                small_change_pixel[j][0] = 0.5
                small_change_pixel[j][1] = 0
                small_change_pixel[j][2] = 0

                small_perturbation_pixel[j][0] = 0.5
                small_perturbation_pixel[j][1] = 0
                small_perturbation_pixel[j][2] = 0

                break


    small_change_pixel = np.reshape(small_change_pixel, (28, 28, 3))
    small_perturbation_pixel = np.reshape(small_perturbation_pixel, (28, 28, 3))

    for i in range(30):
        for j in range(784):

            if extraion_arr_reshape[j] == select_big[i]:

                if big_change_pixel[j][1] == 0.5 and big_perturbation_pixel[j][1] == 0.5:
                    continue

                big_change_pixel[j][0] = 0
                big_change_pixel[j][1] = 0.5
                big_change_pixel[j][2] = 0

                big_perturbation_pixel[j][0] = 0
                big_perturbation_pixel[j][1] = 0.5
                big_perturbation_pixel[j][2] = 0

                break

    big_change_pixel = np.reshape(big_change_pixel, (28, 28, 3))
    big_perturbation_pixel = np.reshape(big_perturbation_pixel, (28, 28, 3))

    return perturbation_background, small_perturbation_pixel, big_perturbation_pixel, small_change_pixel, big_change_pixel

def highlight_solo_pixel(model, img):
    
    saliency_img = eval('vanilla_saliency')(model, img)

    predict = model.predict(tf.expand_dims(img, 0))
    predict = np.argmax(predict)

    extraion_arr = saliency_img
    extraion_arr_reshape = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshape)

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

            if extraion_arr_reshape[i] == select_small[j]:
                small_change_pixel[i][0] = 0.5
                small_change_pixel[i][1] = 0
                small_change_pixel[i][2] = 0


    small_change_pixel = np.reshape(small_change_pixel, (28, 28, 3))

    # plt.imshow(small_change_pixel)
    # plt.savefig("solo_small_change_pixel.png")

    for i in range(784):
        for j in range(30):

            if extraion_arr_reshape[i] == select_big[j]:
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

    origin_data[0], origin_data[1], origin_data[2], origin_data[3], origin_data[4], origin_data[5], origin_data[6], origin_data[7], origin_data[8], origin_data[9] = x_test[1748], x_test[40], x_test[1790], x_test[734], x_test[227], x_test[1146], x_test[1593], x_test[175], x_test[495], x_test[882]

    if exists(f'./dataset/targeted_cw_data'):
        targeted_cw_data = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

    else:

        for i in range(10):
            for j in range(10):

                targeted_cw_data[i][j] = targeted_cw(model, origin_data[i], j)

                print("드디어 {}의 {} 끝났다.  ".format(i, j))

            pickle.dump(targeted_cw_data, open(f'./dataset/targeted_cw_data','wb'))


    perturbation_cw_data = np.zeros((10, 10, 28, 28, 1)) # perturbation 이미지

    # 새로운 주석
    saliency_origin_data = np.zeros((10, 28, 28, 1)) # 원본 이미지의 saliency map
    saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 1)) # 타겟 이미지의 saliency map

    small_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 작은 픽셀 30개 고르기
    big_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 큰 픽셀 30개 고르기
    abs_small_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 작은 픽셀 30개 고르기
    abs_big_saliency_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 큰 픽셀 30개 고르기

    small_perturbation_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 작은 픽셀 Overlab
    big_perturbation_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 큰 픽셀 Overlab
    abs_small_perturbation_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 작은 픽셀 Overlab
    abs_big_perturbation_targeted_cw_data = np.zeros((10, 10, 28, 28, 3)) # 영향도가 가장 큰 픽셀 Overlab


    perturbation_cw_data = np.zeros((10, 10, 28, 28, 3)) # perturbation 이미지
    
    for i in range(10):
        targeted_cw_data[i][i] = origin_data[i]

    for i in range(10):
        for j in range(10):

            saliency_origin_data[i] = eval('vanilla_saliency')(model, origin_data[i])
            saliency_targeted_cw_data[i][j] = eval('vanilla_saliency')(model, targeted_cw_data[i][j])

            perturbation_cw_data[i][j], small_perturbation_targeted_cw_data[i][j], big_perturbation_targeted_cw_data[i][j], small_saliency_targeted_cw_data[i][j], big_saliency_targeted_cw_data[i][j] = highlight_differnt_saliency_pixel(origin_data[i], targeted_cw_data[i][j], saliency_targeted_cw_data[i][j], saliency_origin_data[i])
            perturbation_cw_data[i][j], abs_small_perturbation_targeted_cw_data[i][j], abs_big_perturbation_targeted_cw_data[i][j], abs_small_saliency_targeted_cw_data[i][j], abs_big_saliency_targeted_cw_data[i][j] = abs_highlight_differnt_saliency_pixel(origin_data[i], targeted_cw_data[i][j], saliency_targeted_cw_data[i][j], saliency_origin_data[i])

    m, n = 10, 7
    fig, axs = plt.subplots(nrows=n, ncols=m, squeeze=True, figsize=(6*m, 6*n))

    for i in range(10):
        for j in range(10):

            axs[0, j].imshow(targeted_cw_data[i][j], cmap="gray")
            axs[0, j].set_title("targeted cw {}".format(j), fontsize=25, fontweight='bold')
            axs[0, j].axis('off')
            
            axs[1, j].imshow(perturbation_cw_data[i][j], cmap="gray")
            axs[1, j].set_title("perturbation {}".format(j), fontsize=25, fontweight='bold')
            axs[1, j].axis('off')

            axs[2, j].imshow(saliency_targeted_cw_data[i][j], cmap="gray")
            axs[2, j].set_title("Saliency cw {}".format(j), fontsize=25, fontweight='bold')
            axs[2, j].axis('off')

            axs[3, j].imshow(small_saliency_targeted_cw_data[i][j], cmap="gray")
            axs[3, j].set_title("(low) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[3, j].axis('off')

            axs[4, j].imshow(small_perturbation_targeted_cw_data[i][j], cmap="gray")
            axs[4, j].set_title("(O low) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[4, j].axis('off')

            axs[5, j].imshow(big_saliency_targeted_cw_data[i][j], cmap="gray")
            axs[5, j].set_title("(High) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[5, j].axis('off')

            axs[6, j].imshow(big_perturbation_targeted_cw_data[i][j], cmap="gray")
            axs[6, j].set_title("(O high) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[6, j].axis('off')

            # axs[7, j].imshow(abs_small_saliency_targeted_cw_data[i][j], cmap="gray")
            # axs[7, j].set_title("(abs low) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            # axs[7, j].axis('off')

            # axs[8, j].imshow(abs_small_perturbation_targeted_cw_data[i][j], cmap="gray")
            # axs[8, j].set_title("(O abs low) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            # axs[8, j].axis('off')

            # axs[9, j].imshow(abs_big_saliency_targeted_cw_data[i][j], cmap="gray")
            # axs[9, j].set_title("(abs High) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            # axs[9, j].axis('off')

            # axs[10, j].imshow(abs_big_perturbation_targeted_cw_data[i][j], cmap="gray")
            # axs[10, j].set_title("(O abs high) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            # axs[10, j].axis('off')

        fig.savefig("./img/{}.png".format(i))
    
    for i in range(10):
        for j in range(10):
            pred = model.predict(tf.expand_dims(targeted_cw_data[i][j], 0))
            pred = np.argmax(pred)

            print("{}   {}    {} ".format(i, j, pred))
        print("-----------------")

def comparision_neuron_activation(model, data):

    ######
    # Hidden layer 1
    ######
    
    for hidden_layer_level in range(len(model.layers)-1):
    # for hidden_layer_level in range(5):
        
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[hidden_layer_level].output)

        for label_count in range(10):

            intermediate_output = intermediate_layer_model(np.expand_dims(data[label_count], 0))

            if len(intermediate_output.shape) == 4:

                intermediate_output = np.reshape(intermediate_output, (intermediate_output.shape[3], intermediate_output.shape[1], intermediate_output.shape[1]))

                for channel_count in range(intermediate_output.shape[0]):

                    if channel_count == 0:
                        part_of_line = intermediate_output[0]
                    else:
                        part_of_line = np.concatenate((part_of_line, intermediate_output[channel_count]), axis=1)

                if label_count == 0:
                    part_of_block = part_of_line
                else:
                    part_of_block = np.concatenate((part_of_block, part_of_line), axis=0)
    
            elif len(intermediate_output.shape) == 2:

                intermediate_output = np.reshape(intermediate_output, (int(np.sqrt(intermediate_output.shape[1])), int(np.sqrt(intermediate_output.shape[1]))) )

                if label_count == 0:
                    part_of_block = intermediate_output
                else:
                    part_of_block = np.concatenate((part_of_block, intermediate_output), axis=0)

        line_draw_position = []

        for i in range(10):
            i += 1
            line_draw_position.append(int(part_of_block.shape[0] / 10) * i)

        x = [0, part_of_block.shape[1]]
        y0 = [line_draw_position[0], line_draw_position[0]]
        y1 = [line_draw_position[1], line_draw_position[1]]
        y2 = [line_draw_position[2], line_draw_position[2]]
        y3 = [line_draw_position[3], line_draw_position[3]]
        y4 = [line_draw_position[4], line_draw_position[4]]
        y5 = [line_draw_position[5], line_draw_position[5]]
        y6 = [line_draw_position[6], line_draw_position[6]]
        y7 = [line_draw_position[7], line_draw_position[7]]
        y8 = [line_draw_position[8], line_draw_position[8]]

        plt.plot(x, y0, 'w', markersize=1)
        plt.plot(x, y1, 'w', markersize=1)
        plt.plot(x, y2, 'w', markersize=1)
        plt.plot(x, y3, 'w', markersize=1)
        plt.plot(x, y4, 'w', markersize=1)
        plt.plot(x, y5, 'w', markersize=1)
        plt.plot(x, y6, 'w', markersize=1)
        plt.plot(x, y7, 'w', markersize=1)
        plt.plot(x, y8, 'w', markersize=1)

        plt.axis('off')
        plt.imshow(part_of_block)
        plt.colorbar()
        plt.savefig("./{}.png".format(hidden_layer_level))
        plt.cla()
        
