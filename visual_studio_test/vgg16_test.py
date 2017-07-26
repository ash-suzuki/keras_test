import os
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.core import Lambda
from keras.applications.vgg16 import VGG16

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, )
x = model.layers[-1].output
x = Lambda(lambda x: K.mean(x, axis=3))(x)
vgg_with_sum = Model(inputs=model.layers[0].input, outputs=x)
vgg_with_sum.summary()