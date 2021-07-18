import tensorflow as tf
import numpy as np

from tqdm import tqdm #아래 두개는 for문 그래프 보여주는 것
from tqdm import trange

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.fgsm_core import fgsm_core_method

import saliency.core as saliency

def mnist_data():

    dataset = tf.keras.datasets.mnist
        
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def model_fn(images, call_model_args, expected_keys=None):
    target_class_idx = call_model_args['class']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output = model(images)
            output = output[:,target_class_idx]
            gradients = np.array(tape.gradient(output, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv, output = model(images)
            gradients = np.array(tape.gradient(output, conv))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

def vanilla_saliency(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(img, model_fn, args)
    attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

def specific_vanilla_saliency(model, img, specific_class):

    args = {'model': model, 'class': specific_class}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(img, model_fn, args)
    attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

def ig(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    ig = saliency.IntegratedGradients()
    attr = ig.GetMask(img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

def untargeted_fgsm(model, img, eps):
    
    img = tf.expand_dims(img, 0)

    fgsm_data = fast_gradient_method(model, img, eps, np.inf)

    return fgsm_data[0]

def targeted_fgsm(model, img, eps, target):
    
    img = tf.expand_dims(img, 0)
    
    target = tf.expand_dims(tf.convert_to_tensor(target, dtype=tf.int64), 0)
    # target = np.array(tf.expand_dims(target, 0))

    fgsm_data = fast_gradient_method(model, img, eps, np.inf, y=target, targeted=True)

    # fgsm_data = fast_gradient_method(model, img, eps, np.inf, targeted=True)

    return fgsm_data[0]

def fgsm_perturbation(model, img, eps):
        
    img = tf.expand_dims(img, 0)

    fgsm_data = fgsm_core_method(model, img, eps, np.inf)

    return fgsm_data[0]



def untargeted_pgd(model, img, eps):

    img = tf.expand_dims(img, 0)

    pgd_data = projected_gradient_descent(model, img, eps, 0.01, 40, np.inf)

    return pgd_data[0]

def targeted_pgd(model, img, eps, target):

    target = tf.expand_dims(tf.convert_to_tensor(target, dtype=tf.int64), 0)
    
    img = tf.expand_dims(img, 0)

    pgd_data = projected_gradient_descent(model, img, eps, 0.01, 40, np.inf, y=target, targeted=True)

    return pgd_data[0]

def untargeted_cw(model, img):

    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    cw_data = carlini_wagner_l2(model, img)

    return cw_data[0]

def targeted_cw(model, img, target):
    
    target = tf.expand_dims(tf.convert_to_tensor(target, dtype=tf.int64), 0)

    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    cw_data = carlini_wagner_l2(model, img, y=target, targeted=True)

    return cw_data[0]


def mim(model, img, eps):
    
    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    mim_data = momentum_iterative_method(model, img, eps)

    return mim_data[0]