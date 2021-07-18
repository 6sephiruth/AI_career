import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import os

# os.environ["CUDA_VISIBLE_DEVICES"]='3'

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)


    model = Sequential([
    Dense(1, input_shape=[1]),
    ])
    
    model.compile(optimizer='sgd', loss='mse')

    model.fit(xs, ys, epochs=1200, verbose=0)
    
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

    print(model.predict([10.0]))