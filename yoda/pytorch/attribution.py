import pickle
import torch
import numpy as np
import time

import saliency.core as saliency
from torchvision import models, transforms
from torch.autograd import Variable

device = 'cuda:2' if torch.cuda.is_available() else 'cuda:3'

def vanilla_saliency(model, img):
    """
    Saliency map을 이용하여 기여도 맵 추출 함수

    :model: 학습된 인공지능 모델
            인공지능 모델이 바뀔 때, 기여도 맵 또한 변경됨.
    :img:   기여도 맵을 추출하고 하는 이미지 데이터
    :return: 추출된 기여도 맵

    """

    pred = model(img.cuda().to(device))

    
    pred = pred.cpu().data.numpy().argmax()

    args = {'model': model, 'class': pred}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(torch.tensor(img).cpu(), model_fn, args)[0]
    attr = np.transpose(attr, (1, 2, 0))
    attr = saliency.VisualizeImageGrayscale(attr)

    return np.reshape(attr, (*attr.shape, 1))


conv_layer_outputs = {}

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    # images = images/255
    # images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    return images.requires_grad_(True)

def model_fn(images, call_model_args, expected_keys=None):

    images = PreprocessImages(images)
    
    target_class_idx =  call_model_args['class']
    model = call_model_args['model']

    output = model(images[0].cuda().to(device))
    m = torch.nn.Softmax(dim=1)
    output = m(output)

    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))


        grads = torch.movedim(grads[0], 1, 2)

        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs
