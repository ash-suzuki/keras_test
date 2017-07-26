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

def load_mnist(npy_dir=os.path.join('C:\\', 'Users', 'ash', 'Downloads', 'mnist_npy')):
    x_train = np.load(os.path.join(npy_dir, 'x_train.npy'))
    x_test = np.load(os.path.join(npy_dir, 'x_test.npy'))
    y_train = np.load(os.path.join(npy_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(npy_dir, 'y_test.npy'))
    return (x_train, y_train), (x_test, y_test)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

batch_size = 128
n_classes = 10
epochs   = 5
n_data    = 28*28
log_filepath = '../log'
npy_dir = os.path.join('C:\\', 'Users', 'ash', 'Downloads', 'mnist_npy')

# load data
print("Loading the data...")
(x_train, y_train), (x_test, y_test) = load_mnist(npy_dir=os.path.join('C:\\', 'Users', 'ash', 'Downloads', 'mnist_npy'))
print("Completed")

# reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# rescale
x_train = x_train.astype(np.float32)
x_test  = x_test.astype(np.float32)
x_train /= 255
x_test  /= 255

# convert class vectors to binary class matrices (one hot vectors)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    # set learning phase parameter (needed if the model is different in the training phase such as Dropout)
    KTF.set_learning_phase(1)
    # build model
    model = Sequential()
    model.add(Dense(64, input_shape=(n_data,), init='normal', name='dense1'))
    model.add(Activation('relu', name='relu1'))
    model.add(Dropout(0.2, name='dropout1'))
    model.add(Dense(64, init='normal', name='dense2'))
    model.add(Activation('relu', name='relu2'))
    model.add(Dropout(0.2, name='dropout2'))
    model.add(Dense(10, init='normal', name='dense3'))
    model.add(Activation('softmax', name='softmax1'))       
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    cbks = [tb_cb]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=cbks, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])

KTF.set_session(old_session)
