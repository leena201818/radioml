from keras.layers import  Embedding
from keras.models import Sequential
import numpy as np

model = Sequential()

model.add(Embedding(input_dim=1000, output_dim=64,input_length=10))
# input_dim:字典表长度
# output_dim:嵌入表示的向量长度(dense数字表示的字映射成向量)
# input_length:输入序列的长度，其中每一个元素为一个字(dense数字表示)
# Embedding层输入参数为（batch,input_length），输出参数为(batch,input_length,output_dim)

input_array = np.random.randint(1000, size=(32, 10))

model.compile(optimizer='rmsprop',loss='mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
print(output_array)