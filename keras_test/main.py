import os
import sys
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from tensorboard_environment import TensorboardEnvironment
from load_mnist import load_mnist
from my_model import simple_full_connected, vgg_like

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

batch_size = 128
n_classes = 10
epochs   = 5
n_data    = 28*28
log_filepath = '../log'

# load data
(x_train, y_train), (x_test, y_test) = load_mnist()

with TensorboardEnvironment():
    model = simple_full_connected()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboard_callback], validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])
