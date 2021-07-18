
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

os.environ["CUDA_VISIBLE_DEVICES"]='3'

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale= 1. /255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        validation_split=0.2
    )


    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=128,
                                                        target_size=(150, 150),
                                                        class_mode='categorical',
                                                        subset='training')

    validation_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=128,
                                                        target_size=(150, 150),
                                                        class_mode='categorical',
                                                        subset='validation')

    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dense(216, activation='relu'),
        
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                save_weights_only=True, 
                                save_best_only=True, 
                                monitor='val_loss', 
                                verbose=1)

    model.fit(train_generator,
                    validation_data=(validation_generator),
                    epochs=30,
                    callbacks=[checkpoint],
                   )

    model.load_weights(checkpoint_path)

    print(model.evaluate(validation_generator))
    return model

# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-rps.h5")