import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers


class mni_cnn2(Model):
    def __init__(self):
        super(mni_cnn, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class mni_cnn(Model):
    def __init__(self):
        super(mni_cnn, self).__init__()
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

class mni_dnn(Model):
    def __init__(self):
        super(mni_dnn, self).__init__()
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

