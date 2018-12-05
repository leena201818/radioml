import keras
from keras.layers import Input
import numpy as np

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)
x = Input(shape=(10,),name='input')
model = SimpleMLP()
model.compile(optimizer='adam',loss='binary_crossentropy')
X = np.random.randint(1,1000,[20,10])
Y = np.random.randint(0,1,[20,10])
model.fit(X,Y,batch_size=10,epochs=2,verbose=1,validation_split=0.2,shuffle=True)

model.summary()


