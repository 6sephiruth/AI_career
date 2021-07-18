
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import os

# os.environ["CUDA_VISIBLE_DEVICES"]='3'


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    model = Sequential([
        # Flatten으로 shape 펼치기
        Flatten(input_shape=(28, 28)),
        # Dense Layer
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
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
                    epochs=14,
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
    model.save("TF2-fashion-mnist.h5")
