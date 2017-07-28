from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.engine.topology import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import SGD
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from tensorboard_environment import TensorboardEnvironment
from keras.applications.vgg16 import VGG16

def simple_full_connected(input_shape=(28, 28), n_classes=10):
    input_tensor = Input(shape=input_shape)
    x = Flatten()(input_tensor)
    x = Dense(1024, name='dense1', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
    x = Activation('relu', name='relu1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = Dropout(0.0, name='dropout1')(x)
    x = Dense(1024, name='dense2', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
    x = Activation('relu', name='relu2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = Dropout(0.0, name='dropout2')(x)
    x = Dense(n_classes, name='dense3', kernel_initializer='normal') (x)
    x = Activation('softmax', name='softmax1')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

def vgg_like(input_shape=(28, 28), n_classes=10):
    input_tensor = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model
