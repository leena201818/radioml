"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax
from keras.layers import LSTM


def ConvBNReluUnit(input,kernel_size = 8,index = 0):
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu',kernel_initializer='glorot_uniform',
               name='conv{}'.format(index + 1))(input)
    # x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    x = MaxPool1D(pool_size=2, strides=2, name='maxpool{}'.format(index + 1))(x)
    return x

def CLDNNLikeModel(weights=None,
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

    #Cnvolutional Block
    L = 4
    for i in range(L):
        x = ConvBNReluUnit(x,kernel_size = tap,index=i)

    #LSTM Unit
    # batch_size,64,2
    x = LSTM(units=50,return_sequences = True)(x)
    x = LSTM(units=50)(x)

    #DNN
    x = Dense(128,activation='selu',name='fc1')(x)
    x = Dropout(dr)(x)
    x = Dense(128, activation='selu', name='fc2')(x)
    x = Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = input,outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
if __name__ == '__main__':
    model = CLDNNLikeModel(None,input_shape=(1024,2),classes=24)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())