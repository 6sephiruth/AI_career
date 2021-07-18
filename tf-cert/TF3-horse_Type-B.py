import urllib.request
import zipfile
import numpy as np
from IPython.display import Image

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import os
os.environ["CUDA_VISIBLE_DEVICES"]='2, 3'



import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"

    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')

    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()

    urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')
    local_zip = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/validation-horse-or-human/')
    zip_ref.close()
    class_map = {
        0: 'Horse',
        1: 'Human', 
    }

    original_datagen = ImageDataGenerator(rescale=1./255)
    original_generator = original_datagen.flow_from_directory('tmp/horse-or-human/', 
                                                            batch_size=128, 
                                                            target_size=(300, 300), 
                                                            class_mode='categorical'
                                                            )

    TRAINING_DIR = 'tmp/horse-or-human/'
    VALIDATION_DIR = 'tmp/validation-horse-or-human/'
    training_datagen = ImageDataGenerator(
        rescale=1/ 255.0,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        fill_mode='nearest'
        
    )
    validation_datagen = ImageDataGenerator(
        rescale=1/ 255.0,
    )
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary',
    )
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary',
    )

    model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(4, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    checkpoint_path = "tmp_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                save_weights_only=True, 
                                save_best_only=True, 
                                monitor='val_loss', 
                                verbose=1)
    model.fit(train_generator, 
          validation_data=(validation_generator),
          epochs=25,
          callbacks=[checkpoint],
          )


    model.load_weights(checkpoint_path)

    print(model.evaluate(validation_generator))
    return model


if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-horses-or-humans-type-B.h5")