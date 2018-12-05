import keras
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

np.random.seed(2018)

'''通过推特的标题和发表日期辅助预测点击数量（多输入、多任务回归模型）'''
def TwitterHitModel():
    # 第一个输入：推特标题，以１００个字符的序列表示
    # 每个数字在１－１００００之间，每个数字代表一个词，可采用１００００维向量表示，１００００是字典大小
    main_input = Input(shape=(100,), dtype='int32', name='main_input')

    #嵌入层：将１００００维的向量嵌入到５１２维词向量，Embdeding作为网络第一层，需要知道序列的长度是１００
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

    # 采用一个LSTM层将标题序列转换成一个特征向量，return_sequences=False，该特征向量是最后一个状态向量，包含了整个输入序列的信息
    lstm_out = LSTM(32, name='LSTM')(x)

    #第一个输出：辅助，仅通过标题序列预测点击数量
    auxiliary_output = Dense(units=1,activation='sigmoid',name='aux_output')(lstm_out)

    #第二个输入：辅助输推测，推特发文时间　month,day,hour,minute,second
    auxiliary_input = Input(shape=(5,),name='aux_input')

    #将辅助输入和ＬＳＴＭ输出拼接在一起
    x = keras.layers.concatenate(inputs=[lstm_out,auxiliary_input])

    #堆砌全连接网络
    x = Dense(units=64,activation='relu',name='dense1')(x)
    x = Dense(units=64,activation='relu',name='dense2')(x)
    x = Dense(units=64,activation='relu',name='dense3')(x)

    #最后一个回归层
    main_output = Dense(units=1,activation='sigmoid',name='main_output')(x)

    #定义一个两个输入、两个输出的网络
    model = Model(inputs=[main_input,auxiliary_input], outputs=[main_output,auxiliary_output])

    #编译模型：辅助输出的权重为0.2,主输出的权重为１，定义损失函数，优化方法
    model.compile(optimizer='rmsprop',loss={'main_output':'binary_crossentropy','aux_output':'binary_crossentropy'},
                  metrics={'main_output': 'accuracy','aux_output':'accuracy'},
                  loss_weights={'main_output':1,'aux_output':0.2})

    return model

def train():
    #生成样本数据
    n_samples = 5000
    headline_data = np.random.randint(1,10000, size=(n_samples,100))
    additional_data = np.random.randint(1,10000, size=(n_samples,5))
    # labels = np.random.randint(1,500,size=(n_samples,1))
    labels = np.sum(additional_data,axis=1)

    model = TwitterHitModel()
    print(model.summary())

    model.fit(x=[headline_data,additional_data],y=[labels,labels],shuffle=True,epochs=20,batch_size=1000)

#获取中间层输出
def intermediate_layer_model(model):
    n_samples = 50
    headline_data = np.random.randint(1, 10000, size=(n_samples, 100))
    additional_data = np.random.randint(1, 10000, size=(n_samples, 5))
    labels = np.random.randint(1, 500, size=(n_samples, 1))


    intermediate_layer_model = Model(inputs=model.input[0],
                                     outputs=model.get_layer("LSTM").output)
    intermediate_output = intermediate_layer_model.predict(headline_data)
    print(intermediate_layer_model.summary())

    print(intermediate_output.shape)

    model.fit(x={'main_input':headline_data,'aux_input':additional_data},y={'main_output':labels,'aux_output':labels},
              epochs=7, batch_size=32)

if __name__ == '__main__':
    train()
    model = TwitterHitModel()
    # intermediate_layer_model(model)
    print('completed!')