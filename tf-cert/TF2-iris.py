import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import os

# os.environ["CUDA_VISIBLE_DEVICES"]='3'

def preprocess(data):
    # YOUR CODE HERE
    # Should return features and one-hot encoded labels

    x = data['features']
    y = data['label']
    y = tf.one_hot(y, 3)

    return x, y

def solution_model():

    train_dataset = tfds.load('iris', split='train[:80%]')
    valid_dataset = tfds.load('iris', split='train[80%:]')
    
    batch_size=10
    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)
    # YOUR CODE TO TRAIN A MODEL HERE

    model = tf.keras.models.Sequential([
        # input_shape는 X의 feature 갯수가 4개 이므로 (4, )로 지정합니다.
        Dense(512, activation='relu', input_shape=(4,)),\
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),

        # Classification을 위한 Softmax, 클래스 갯수 = 3개
        Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                save_weights_only=True, 
                                save_best_only=True, 
                                monitor='val_loss', 
                                verbose=1)

    model.fit(train_data,
                    validation_data=(valid_data),
                    epochs=20,
                    callbacks=[checkpoint],
                   )
    # checkpoint 를 저장한 파일명을 입력합니다.
    model.load_weights(checkpoint_path)

    print(model.evaluate(valid_data))
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-iris.h5")