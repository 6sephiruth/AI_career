import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers

from keras.applications.vgg16 import VGG16

class mnist_cnn2(Model):
    def __init__(self):
        super(mnist_cnn2, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class mnist_cnn(Model):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3),
                activation='relu',
                padding='same',
                input_shape=(28, 28, 1)))
        model.add(MaxPool2D((2, 2)))

        model.add(Conv2D(64, (3, 3),
                activation='relu'))

        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3),
                activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)

class mnist_dnn(Model):
    def __init__(self):
        super(mnist_dnn, self).__init__()
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        # Flatten으로 shape 펼치기
        model.add(Flatten(input_shape=(28, 28)))
        # Dense Layer
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # Classification을 위한 Softmax 
        model.add(Dense(10, activation='softmax'))
    
        return model
        
    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)

class cifar_vgg16(Model):

    def __init__(self):
        super(cifar_vgg16, self).__init__()
        self.model = self.build_model()
    
    def build_model(self):

        transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        transfer_model.trainable = False

        model = Sequential()
        model.add(transfer_model)
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Reshape((784,), input_shape=(28, 28, 1)),
            layers.Dense(128, activation="relu"),
            
            layers.Dense(32, activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),                        
            layers.Dense(784, activation="sigmoid"),
            layers.Reshape((28,28,1))
        
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# simple autoencoder
class ae2(Model):
    def __init__(self, latent_dim):
        super(ae2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            Flatten(),
            Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
            Dense(784, activation='sigmoid'),
            Reshape((28, 28, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
