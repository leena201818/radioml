"""VGGLike model for RadioML.

# Reference:

- [V ERY D EEP C ONVOLUTIONAL N ETWORKS FOR L ARGE -S CALE I MAGE R ECOGNITION](
    https://arxiv.org/abs/1409.1556)

Adapted from code contributed by Mika.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Flatten
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.normalization import  BatchNormalization
from keras.layers.core import Activation

import os
import warnings

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

# Build VGG-Like Neural Net model using Keras primitives --
#  - Input_Shape is [N,1024,1]
#  - Pass through 7 1DConv/ReLu layers
#  - Pass through 2 Dense layers (SeLu and Softmax)
#  - Perform categorical cross entropy optimization
def VGGLikeModel(weights='ResNetLike-125k.wts.h5',
             input_shape=[1024,2],
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    model = models.Sequential()
    tap = 8

    model.add(Conv1D(input_shape=input_shape, filters=64, kernel_size=tap, padding='same', activation="relu", name="conv1",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Conv1D(filters=64, kernel_size=tap, padding='same', activation="relu", name="conv2",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Conv1D(filters=64, kernel_size=tap, padding='same', activation="relu", name="conv3",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Conv1D(filters=64, kernel_size=tap, padding='same', activation="relu", name="conv4",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Conv1D(filters=64, kernel_size=tap, padding='same', activation="relu", name="conv5",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Conv1D(filters=64, kernel_size=tap, padding='same', activation="relu", name="conv6",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Conv1D(filters=64, kernel_size=tap, padding='same', activation="relu", name="conv7",
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling1D(pool_size=2,strides = 2))

    model.add(Flatten())

    model.add(Dense(128, activation='selu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))

    model.add(Dense(128, activation='selu', kernel_initializer='he_normal', name="dense2"))
    model.add(Dropout(dr))

    model.add(Dense(classes, kernel_initializer='he_normal', name="dense3"))
    model.add(Activation('softmax'))

    # model.add(Reshape((classes,)))

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,BatchNormalization,ReLU,Dropout

def ConvBNReluUnit(input,kernel_size = 8,index = 0):
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', kernel_initializer='glorot_uniform',
               name='conv{}'.format(index + 1))(input)
    x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    x = ReLU(name='conv{}-relu'.format(index + 1))(x)
    x = MaxPool1D(pool_size=2, strides=2, name='maxpool{}'.format(index + 1))(x)
    return x

def VGGLikeModel_with_bn(weights=None,
             input_shape=[1024,2],
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    input = Input(input_shape,name='input')
    x = input

    L = 7
    for i in range(L):
        x = ConvBNReluUnit(x,kernel_size = tap,index=i)

    x = Flatten(name='flatten')(x)
    x = Dense(units = 128,activation='selu', kernel_initializer='he_normal', name='fc1')(x)
    x = Dropout(rate = dr,name='dropout1')(x)
    x = Dense(units=128, activation='selu', kernel_initializer='he_normal', name='fc2')(x)
    x = Dropout(rate=dr, name='dropout2')(x)
    x = Dense(units=classes,activation='softmax',kernel_initializer='he_normal',name='softmax')(x)

    model = Model(inputs = input,outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

