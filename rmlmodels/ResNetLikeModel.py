"""ResNetLike model for RadioML.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

Adapted from code contributed by Mika.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers
from keras import models

import os
import warnings

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')


def residual_unit(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 2 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    bn_axis = 2

    conv_name_base = 'res_stack' + str(stage) + '_block'+block
    bn_name_base = 'bn_stack' + str(stage) + '_block'+block

    x = layers.Conv1D(filters=filters1, kernel_size=kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '_u1')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_u1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=filters2, kernel_size=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '_u2')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_u2')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def residual_stack(input_tensor,
                   filters,
                   stage,
                   strides=2):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1 = filters
    bn_axis = 2

    conv_name_base = 'res_stack' + str(stage) + '_'
    bn_name_base = 'bn_stack' + str(stage) + '_'
    mp_name_base = 'mp_stack' + str(stage)

    x = layers.Conv1D(filters1, 1, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)

    taps = 8
    x = residual_unit(input_tensor=x,kernel_size=taps,filters=[32,32],stage=stage,block='b')
    x = residual_unit(input_tensor=x, kernel_size=taps, filters=[32, 32], stage=stage, block='c')

    x = layers.MaxPooling1D(pool_size=3,strides=1,padding='same',name=mp_name_base)(x)

    return x

def ResNetLikeModel(weights='ResNetLike-125k.wts.h5',
             input_shape=None,
             classes=24,
             **kwargs):
    """Instantiates the ResNetLike radioml architecture.
    # Arguments
        weights: one of `None` (random initialization),
              or the path to the weights file to be loaded.
        input_shape: the input shape
            has to be `(batch,1024,2)` (with `channels_last` data format)
        classes: optional number of classes to classify images

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    if classes != 24:
        raise ValueError('classes should be 24')

    img_input = layers.Input(shape=input_shape,name='Input')
    x = img_input

    L = 6
    for i in range(L):
        x = residual_stack(input_tensor = x,filters=32,stage = i+1,strides=2)

    x = layers.Flatten()(x)
    dr = 0.5

    x = layers.Dense(128, activation='selu', kernel_initializer='he_normal', name="dense1")(x)
    x = layers.Dropout(dr)(x)

    x = layers.Dense(128, activation='selu', kernel_initializer='he_normal', name="dense2")(x)
    x = layers.Dropout(dr)(x)

    x = layers.Dense(classes, kernel_initializer='he_normal', name="dense3")(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(img_input, x, name='resnet-like')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

def ResNetLSTMLikeModel(weights='ResNetLike-125k.wts.h5',
             input_shape=None,
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    if classes != 24:
        raise ValueError('classes should be 24')

    img_input = layers.Input(shape=input_shape,name='Input')
    x = img_input

    L = 6
    for i in range(L):
        x = residual_stack(input_tensor = x,filters=32,stage = i+1,strides=2)

    x = layers.LSTM(50)(x)
    dr = 0.5

    x = layers.Dense(128, activation='selu', kernel_initializer='he_normal', name="dense1")(x)
    x = layers.Dropout(dr)(x)

    x = layers.Dense(128, activation='selu', kernel_initializer='he_normal', name="dense2")(x)
    x = layers.Dropout(dr)(x)

    x = layers.Dense(classes, kernel_initializer='he_normal', name="dense3")(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(img_input, x, name='resnet-like')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model