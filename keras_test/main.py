import os
import sys
import numpy as np
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks
import tensorflow as tf
from tensorboard_environment import TensorboardEnvironment
from load_mnist import load_mnist
from my_model import simple_full_connected, vgg_like

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

(x_train, y_train), (x_test, y_test) = load_mnist(x_style='one_channel')
batch_size = 128
n_classes = 10
epochs   = 5
log_filepath = '../log'

with TensorboardEnvironment():
    model = simple_full_connected(input_shape=x_train.shape[1:], n_classes=n_classes)
    model.summary()
    optimizer = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboard_callback], validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])
