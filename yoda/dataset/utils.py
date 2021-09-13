import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def mkdir(dir_names):
    for d in dir_names:
        if not os.path.exists(d):
            os.mkdir(d)

def preprocess(img, lbl):
    _img = tf.cast(img, tf.float32)
    _img = tf.divide(_img, 255)
    _img = tf.image.resize(_img, (224, 224))  # 이미지 사이즈를 얜 224, 아래는 244로 해둬서 한참해멤...ㅋㅋ

    return _img, lbl

def mnist():
    
    dataset = tf.keras.datasets.mnist
        
    train, test = dataset.load_data()

    return train, test

def cifar10():
    
    dataset = tf.keras.datasets.cifar10

    train, test = dataset.load_data()

    return train, test

def cifar100():
    
    dataset = tf.keras.datasets.cifar100

    train, test = dataset.load_data()

    return train, test

def fashion_mnist():

    dataset = tf.keras.datasets.fashion_mnist

    train, test = dataset.load_data()
    
    return train, test

def imdb():
    
    dataset = tf.keras.datasets.imdb
    train, test = dataset.load_data()
    
    return train, test

def boston_housing():
    
    dataset = tf.keras.datasets.boston_housing
    train, test = dataset.load_data()
    
    return train, test

def reuters_newswire():
    
    dataset = tf.keras.datasets.reuters
    train, test = dataset.load_data()
    
    return train, test


def cats_vs_dogs():
    
    train, train2, test = tfds.load(name = 'cats_vs_dogs',
                            split = ('train[:50%]', 'train[50%:80%]', 'train[80%:]'))

    x_train = []
    y_train = []

    x_train2 = []
    y_train2 = []

    x_test = []
    y_test = []

    for data in train:

        image, label = data['image'], data['label']
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255)
        image = tf.image.resize(image, (224, 224))

        x_train.append(image)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in train2:
        
        image, label = data['image'], data['label']
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255)
        image = tf.image.resize(image, (224, 224))

        x_train2.append(image)
        y_train2.append(label)

    x_train2 = np.array(x_train2)
    y_train2 = np.array(y_train2)

    x_train = np.concatenate([x_train, x_train2])
    y_train = np.concatenate([y_train, y_train2])

    for data in test:
        
        image, label = data['image'], data['label']
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255)
        image = tf.image.resize(image, (224, 224))

        x_test.append(image)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def lfw():

    train, test = tfds.load(name = 'lfw',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        image, label = data['image'], data['label']

        x_train.append(image)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        image, label = data['image'], data['label']

        x_test.append(image)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test


def iris():

    train, test = tfds.load(name = 'iris',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features, label = data['features'], data['label']

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features, label = data['features'], data['label']

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test
def iris():

    train, test = tfds.load(name = 'iris',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features, label = data['features'], data['label']

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features, label = data['features'], data['label']

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def emnist():
    
    train, test = tfds.load(name = 'emnist',
                            split = ('train','test'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features, label = data['image'], data['label']

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features, label = data['image'], data['label']

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def kmnist():
    
    train, test = tfds.load(name = 'kmnist',
                            split = ('train','test'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features, label = data['image'], data['label']

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features, label = data['image'], data['label']

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def smallnorb():
    
    train, test = tfds.load(name = 'smallnorb',
                            split = ('train','test'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features, label = data['image'], data['instance']

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features, label = data['image'], data['instance']

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def wine_quality():
    
    train, test = tfds.load(name = 'wine_quality',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features0 = data['features']['alcohol'].numpy()
        features1 = data['features']['chlorides'].numpy()
        features2 = data['features']['citric acid'].numpy()
        features3 = data['features']['density'].numpy()
        features4 = data['features']['fixed acidity'].numpy()
        features5 = data['features']['free sulfur dioxide'].numpy()
        features6 = data['features']['pH'].numpy()
        features7 = data['features']['residual sugar'].numpy()
        features8 = data['features']['sulphates'].numpy()
        features9 = data['features']['total sulfur dioxide'].numpy()
        features10 = data['features']['volatile acidity'].numpy()
        
        label = data['quality']
        
        sum_features = np.stack((features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10), axis=0)

        features = list(sum_features)

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features0 = data['features']['alcohol'].numpy()
        features1 = data['features']['chlorides'].numpy()
        features2 = data['features']['citric acid'].numpy()
        features3 = data['features']['density'].numpy()
        features4 = data['features']['fixed acidity'].numpy()
        features5 = data['features']['free sulfur dioxide'].numpy()
        features6 = data['features']['pH'].numpy()
        features7 = data['features']['residual sugar'].numpy()
        features8 = data['features']['sulphates'].numpy()
        features9 = data['features']['total sulfur dioxide'].numpy()
        features10 = data['features']['volatile acidity'].numpy()
        
        label = data['quality']
        
        sum_features = np.stack((features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10), axis=0)

        features = list(sum_features)

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def titanic():
    
    train, test = tfds.load(name = 'titanic',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for data in train:
        
        features0 = data['features']['age'].numpy()
        features1 = data['features']['boat'].numpy()
        features2 = data['features']['body'].numpy()
        features3 = data['features']['cabin'].numpy()
        features4 = data['features']['embarked'].numpy()
        features5 = data['features']['fare'].numpy()
        features6 = data['features']['home.dest'].numpy()
        features7 = data['features']['name'].numpy()
        features8 = data['features']['parch'].numpy()
        features9 = data['features']['pclass'].numpy()
        features10 = data['features']['sex'].numpy()
        features11 = data['features']['sibsp'].numpy()
        features12 = data['features']['ticket'].numpy()

        label = data['survived']
        
        sum_features = np.stack((features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10, features11, features12), axis=0)

        features = list(sum_features)

        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for data in test:
        
        features0 = data['features']['age'].numpy()
        features1 = data['features']['boat'].numpy()
        features2 = data['features']['body'].numpy()
        features3 = data['features']['cabin'].numpy()
        features4 = data['features']['embarked'].numpy()
        features5 = data['features']['fare'].numpy()
        features6 = data['features']['home.dest'].numpy()
        features7 = data['features']['name'].numpy()
        features8 = data['features']['parch'].numpy()
        features9 = data['features']['pclass'].numpy()
        features10 = data['features']['sex'].numpy()
        features11 = data['features']['sibsp'].numpy()
        features12 = data['features']['ticket'].numpy()

        label = data['survived']
        
        sum_features = np.stack((features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10, features11, features12), axis=0)

        features = list(sum_features)

        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train, test = [x_train, y_train], [x_test, y_test]

    return train, test

def cherry_blossoms():
    
    train, test = tfds.load(name = 'cherry_blossoms',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []

    x_test = []

    for data in train:
        
        features0 = data['doy'].numpy()
        features1 = data['temp'].numpy()
        features2 = data['temp_lower'].numpy()
        features3 = data['temp_upper'].numpy()
        features4 = data['year'].numpy()

        
        sum_features = np.stack((features0, features1, features2, features3, features4), axis=0)

        features = list(sum_features)

        x_train.append(features)
        
    x_train = np.array(x_train)

    for data in test:
        
        features0 = data['doy'].numpy()
        features1 = data['temp'].numpy()
        features2 = data['temp_lower'].numpy()
        features3 = data['temp_upper'].numpy()
        features4 = data['year'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3, features4), axis=0)

        features = list(sum_features)

        x_test.append(features)

    x_test = np.array(x_test)

    train, test = [x_train], [x_test]

    return train, test

def howell():
    
    train, test = tfds.load(name = 'howell',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []

    x_test = []

    for data in train:
        
        features0 = data['age'].numpy()
        features1 = data['height'].numpy()
        features2 = data['male'].numpy()
        features3 = data['weight'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3), axis=0)

        features = list(sum_features)

        x_train.append(features)
        
    x_train = np.array(x_train)

    for data in test:
        
        features0 = data['age'].numpy()
        features1 = data['height'].numpy()
        features2 = data['male'].numpy()
        features3 = data['weight'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3), axis=0)

        features = list(sum_features)

        x_test.append(features)

    x_test = np.array(x_test)

    train, test = [x_train], [x_test]

    return train, test


def blimp():
    
    train, test = tfds.load(name = 'blimp',
                            split = ('train[:80%]','train[80%:]'))

    x_train = []

    x_test = []

    for data in train:
        
        features0 = data['UID'].numpy()
        features1 = data['field'].numpy()
        features2 = data['lexically_identical'].numpy()
        features3 = data['linguistics_term'].numpy()
        features4 = data['one_prefix_method'].numpy()
        features5 = data['pair_id'].numpy()
        features6 = data['sentence_bad'].numpy()
        features7 = data['sentence_good'].numpy()
        features8 = data['simple_LM_method'].numpy()
        features9 = data['two_prefix_method'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3, features4, features5, features6, features7, features8, features9), axis=0)

        features = list(sum_features)

        x_train.append(features)

    x_train = np.array(x_train)

    for data in test:
        
        features0 = data['UID'].numpy()
        features1 = data['field'].numpy()
        features2 = data['lexically_identical'].numpy()
        features3 = data['linguistics_term'].numpy()
        features4 = data['one_prefix_method'].numpy()
        features5 = data['pair_id'].numpy()
        features6 = data['sentence_bad'].numpy()
        features7 = data['sentence_good'].numpy()
        features8 = data['simple_LM_method'].numpy()
        features9 = data['two_prefix_method'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3, features4, features5, features6, features7, features8, features9), axis=0)

        features = list(sum_features)

        x_test.append(features)

    x_test = np.array(x_test)

    train, test = [x_train] , [x_test]

    return train, test

def clinc_oos():
    
    train, test = tfds.load(name = 'clinc_oos',
                            split = ('train', 'test'))
    x_train = []

    x_test = []

    for data in train:
        
        features0 = data['domain'].numpy()
        features1 = data['domain_name'].numpy()
        features2 = data['intent'].numpy()
        features3 = data['intent_name'].numpy()
        features4 = data['text'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3, features4), axis=0)

        features = list(sum_features)

        x_train.append(features)

    x_train = np.array(x_train)

    for data in test:
        
        features0 = data['domain'].numpy()
        features1 = data['domain_name'].numpy()
        features2 = data['intent'].numpy()
        features3 = data['intent_name'].numpy()
        features4 = data['text'].numpy()
        
        sum_features = np.stack((features0, features1, features2, features3, features4), axis=0)

        features = list(sum_features)

        x_test.append(features)

    x_test = np.array(x_test)

    train, test = [x_train] , [x_test]

    return train, test
