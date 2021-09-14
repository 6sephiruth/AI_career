import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torchvision import datasets, transforms


from models import *
from utils import *
from ad_attack import *
from attribution import *


from tqdm import trange

import pickle
import time

device = 'cuda:2' if torch.cuda.is_available() else 'cuda:3'

def highlight_differnt_saliency_pixel(origin_img, targeted_cw_img, saliency_adv_img, saliency_origin_img):
    
    # 방법 1 - 그냥 빼기
    extraion_arr = saliency_adv_img - saliency_origin_img

    extraion_arr_reshape = np.reshape(extraion_arr, (784))

    data_sort = np.sort(extraion_arr_reshape)

    select_small = data_sort[:30]
    select_big = data_sort[-30:]
    
    origin_img = torch.tensor(origin_img)
    # change_pixel = torch.unsqueeze(origin_img, 0)
    # origin_img = np.reshape(origin_img, (28, 28, 1))

    change_pixel = origin_img.repeat(1, 1, 3)
    change_pixel = change_pixel.numpy()
    change_pixel = np.reshape(change_pixel, (28, 28, 3))
    origin_img = origin_img.numpy()

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

    perturbation_background = torch.tensor(perturbation_cw_data)
    # perturbation_background = np.reshape(perturbation_background, (28, 28, 1))

    perturbation_background = perturbation_background.repeat(1, 1, 3)

    perturbation_background = np.reshape(perturbation_background, (28, 28, 3))

    perturbation_background = perturbation_background.numpy()



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

    origin_img = torch.tensor(origin_img)
    # change_pixel = torch.unsqueeze(origin_img, 0)

    # origin_img = np.reshape(origin_img, (28, 28, 1))

    change_pixel = origin_img.repeat(1, 1, 3)
    change_pixel = change_pixel.numpy()
    change_pixel = np.reshape(change_pixel, (28, 28, 3))
    origin_img = origin_img.numpy()

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


    perturbation_background = torch.tensor(perturbation_cw_data)
    # perturbation_background = np.reshape(perturbation_background, (28, 28, 1))

    perturbation_background = perturbation_background.repeat(1, 1, 3)

    perturbation_background = np.reshape(perturbation_background, (28, 28, 3))

    perturbation_background = perturbation_background.numpy()

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

def cw_saliency_analysis(model):

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    dataset2 = datasets.MNIST('../dataset', train=False,
                        transform=transform)


    test_kwargs = {'batch_size': 1}
    test_data = torch.utils.data.DataLoader(dataset2,**test_kwargs)

    # targeted CW dataset 만들기
    origin_data = np.zeros((10, 1, 28, 28))
    targeted_cw_data = np.zeros((10, 10, 1, 28, 28))

    record_cw = np.zeros(10, dtype='int')
    if exists(f'./dataset/targeted_cw_data'):
        targeted_cw_data = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

    else:
        print("targeted cw 데이터에 적합한 데이터 검색 중")

        for i in range(10):
            for batch_idx, (data, target) in enumerate(test_data):

                if target.numpy()[0] == i:
                    
                    for j in range(10):
                            
                        data = data.type(torch.cuda.FloatTensor).to(device)
                        
                        cw_result = targeted_cw(model, data, j)
                        pred = model(cw_result).cpu().data.numpy().argmax()

                        if pred != j:
                            break
                        record_cw[i] = batch_idx

                if record_cw[i] != 0:
                    break

        for batch_idx, (data, target) in enumerate(test_data):

            if batch_idx == record_cw[0]: origin_data[0] = data
            if batch_idx == record_cw[1]: origin_data[1] = data
            if batch_idx == record_cw[2]: origin_data[2] = data
            if batch_idx == record_cw[3]: origin_data[3] = data
            if batch_idx == record_cw[4]: origin_data[4] = data
            if batch_idx == record_cw[5]: origin_data[5] = data
            if batch_idx == record_cw[6]: origin_data[6] = data
            if batch_idx == record_cw[7]: origin_data[7] = data
            if batch_idx == record_cw[8]: origin_data[8] = data
            if batch_idx == record_cw[9]: origin_data[9] = data
            


        print("targeted attack 데이터 생성 중")

        for i in range(10):
            for j in range(10):

                img = torch.unsqueeze(torch.tensor(origin_data[i]).to(device), 0)
                img = img.type(torch.cuda.FloatTensor)

                cw_data =  targeted_cw(model, img, j)
                cw_data = cw_data.cpu()
                
                targeted_cw_data[i][j] = cw_data[0]

                print("정상 label {}를 적대적 예제 {}로 변환. ".format(i, j))

            pickle.dump(targeted_cw_data, open(f'./dataset/targeted_cw_data','wb'))


    # perturbation_cw_data = np.zeros((10, 10, 28, 28, 1)) # perturbation 이미지

    # # 새로운 주석
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
            
            origin_img = torch.unsqueeze(torch.tensor(origin_data[i]), 0)
            origin_img = origin_img.type(torch.cuda.FloatTensor)
            saliency_origin_data[i] = eval('vanilla_saliency')(model, origin_img)

            targeted_img = torch.unsqueeze(torch.tensor(targeted_cw_data[i][j]), 0)
            targeted_img = targeted_img.type(torch.cuda.FloatTensor)

            saliency_targeted_cw_data[i][j] = eval('vanilla_saliency')(model, targeted_img)

            perturbation_cw_data[i][j], small_perturbation_targeted_cw_data[i][j], big_perturbation_targeted_cw_data[i][j], small_saliency_targeted_cw_data[i][j], big_saliency_targeted_cw_data[i][j] = highlight_differnt_saliency_pixel(np.reshape(origin_data[i], (28, 28, 1)), np.reshape(targeted_cw_data[i][j], (28, 28, 1)), saliency_targeted_cw_data[i][j], saliency_origin_data[i])
            perturbation_cw_data[i][j], abs_small_perturbation_targeted_cw_data[i][j], abs_big_perturbation_targeted_cw_data[i][j], abs_small_saliency_targeted_cw_data[i][j], abs_big_saliency_targeted_cw_data[i][j] = abs_highlight_differnt_saliency_pixel(np.reshape(origin_data[i], (28, 28, 1)), np.reshape(targeted_cw_data[i][j], (28, 28, 1)), saliency_targeted_cw_data[i][j], saliency_origin_data[i])

    targeted_cw_data = np.reshape(targeted_cw_data, (10, 10, 28, 28, 1))

    m, n = 10, 11
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

            axs[7, j].imshow(abs_small_saliency_targeted_cw_data[i][j], cmap="gray")
            axs[7, j].set_title("(abs low) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[7, j].axis('off')

            axs[8, j].imshow(abs_small_perturbation_targeted_cw_data[i][j], cmap="gray")
            axs[8, j].set_title("(O abs low) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[8, j].axis('off')

            axs[9, j].imshow(abs_big_saliency_targeted_cw_data[i][j], cmap="gray")
            axs[9, j].set_title("(abs High) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[9, j].axis('off')

            axs[10, j].imshow(abs_big_perturbation_targeted_cw_data[i][j], cmap="gray")
            axs[10, j].set_title("(O abs high) SA - SO {}".format(j), fontsize=25, fontweight='bold')
            axs[10, j].axis('off')

        fig.savefig("./img/{}.png".format(i))