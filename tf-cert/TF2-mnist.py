# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the MNIST dataset
# Note that the test will expect it to classify 10 classes and that the 
# input shape should be the native size of the MNIST dataset which is 
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#


import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import os

# os.environ["CUDA_VISIBLE_DEVICES"]='3'

def solution_model():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    model = Sequential([
        # Flatten으로 shape 펼치기
        Flatten(input_shape=(28, 28)),
        # Dense Layer
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        # Classification을 위한 Softmax 
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                save_weights_only=True, 
                                save_best_only=True, 
                                monitor='val_loss', 
                                verbose=1)

    model.fit(x_train, y_train,
                validation_data=(x_valid, y_valid),
                epochs=10,
                callbacks=[checkpoint],
                )

    model.load_weights(checkpoint_path)

    print(model.evaluate(x_valid, y_valid))

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-mnist.h5")
