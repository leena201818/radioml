import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt
import sys, keras
import h5py

from keras.regularizers import *
from keras.optimizers import adam
import keras.backend as K
from keras.callbacks import LearningRateScheduler

import mltools
import rmldataset2016
import rmlmodels.ResNetLikeModel as resnet
import rmlmodels.VGGLikeModel as vggnet
import rmlmodels.CLDNNLikeModel as cldnn
from keras.models import model_from_json

import warnings
warnings.filterwarnings("ignore")


classes = ['32PSK',
           '16APSK',
           '32QAM',
           'FM',
           'GMSK',
           '32APSK',
           'OQPSK',
           '8ASK',
           'BPSK',
           '8PSK',
           'AM-SSB-SC',
           '4ASK',
           '16PSK',
           '64APSK',
           '128QAM',
           '128APSK',
           'AM-DSB-SC',
           'AM-SSB-WC',
           '64QAM',
           'QPSK',
           '256QAM',
           'AM-DSB-WC',
           'OOK',
           '16QAM']

def train(from_filename = '/media/norm_XYZ_1024_128k.hdf5',weight_file='weights/norm_res-like-128k.wts.h5',init_weight_file=None):

    f = h5py.File(from_filename, 'r')  # 打开h5文件
    X = f['X'][:,:,:]  # ndarray(2555904*512*2)
    Y = f['Y'][:,:]  # ndarray(2M*24)
    Z = f['Z'][:]  # ndarray(2M*1)

    # [N,1024,2]
    in_shp = X[0].shape

    n_examples = X.shape[0]
    n_train = int(n_examples * 0.8)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    # Set up some params
    nb_epoch = 125     # number of epochs to train on
    batch_size = 1024  # training batch size

    # perform training ...
    #   - call the main training loop in keras for our network+dataset

    # model = resnet.ResNetLikeModel(None,input_shape=X[0].shape,classes=24)
    model = vggnet.VGGLikeModel(None, input_shape=X[0].shape, classes=24)
    # model = vggnet.VGGLikeModel_with_bn(None,input_shape=X[0].shape, classes=24)
    # model = cldnn.CLDNNLikeModel(None,input_shape=X[0].shape,classes=24)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())

    # with open('models/cldnn.json','w') as f:
    #     f.write(model.to_json())

    return

    def scheduler(epoch):
        if int(epoch % 80) == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {} at epoch{}".format(lr * 0.5,epoch))
            return K.get_value(model.optimizer.lr)
        else:
            print("epoch({}) lr is {}".format(epoch,K.get_value(model.optimizer.lr)))
            return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    if init_weight_file is not None:
        model.load_weights(init_weight_file)

    history = model.fit(X_train,
        Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks = [reduce_lr,
                     keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                     # keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                    ]
                        )

    # we re-load the best weights once training is finished
    # model.load_weights(weight_file)
    mltools.show_history(history)

    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print(score)

    f.close()

def predict(model,weight_file='res-like-1024-1m.wts.h5',test_filename = '/media/XYZ_1024_128k.hdf5',dis_acc=True,dis_conf=True,min_snr = 0):
    test_file = h5py.File(test_filename, 'r')
    X = test_file['X']
    Y = test_file['Y'][:]
    Z = test_file['Z'][:]

    global classes
    model.load_weights(weight_file)
    Y_hat   =   model.predict(X,batch_size=1024,verbose=1)

    test_file.close()

    #plot confusion matrix
    cm, right, wrong = mltools.calculate_confusion_matrix(Y,Y_hat,classes)
    acc = round(1.0 * right / (right + wrong),4)
    print('Overall Accuracy:%.2f%s / (%d + %d)'%(100*acc,'%',right, wrong))

    if dis_conf:
        mltools.plot_confusion_matrix(cm,'Confution matrix of {}'.format(test_filename),labels=classes)

    #plot accuracy with erery snr
    if dis_acc:
        print(min_snr)
        mltools.calculate_acc_cm_each_snr(Y,Y_hat,Z,classes,min_snr = min_snr)

def plot_tSNE(model,weight_file='good/vgg-like-1024-1m-tap8.wts-58.9.h5',test_filename = '/media/XYZ_1024_64k.hdf5'):
    from keras.models import Model
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    test_file = h5py.File(test_filename, 'r')
    X = test_file['X'][:]
    Y = test_file['Y'][:]
    Z = test_file['Z'][:]

    global classes
    model.load_weights(weight_file)
    # Y_hat = model.predict(X, batch_size=1024, verbose=1)

    test_file.close()

    #设计中间层输出的模型
    dense1_model = Model(inputs=model.input,outputs=model.get_layer('dense1').output)

    #提取snr下的数据进行测试
    snrs = [30]
    for snr in [s for s in snrs if s > 14]:
        test_X_i = X[np.where(Z == snr)[0],:,:]
        Y_true = Y[np.where(Z == snr)[0],:]
        Y_true_label = np.argmax(Y_true,axis=1)

        #计算中间层输出
        dense1_output = dense1_model.predict(test_X_i,batch_size=32)

        #PCA降维到50以内
        pca = PCA(n_components=30)
        dense1_output_pca = pca.fit_transform(dense1_output)

        #t-SNE降为2
        tsne = TSNE(n_components=2,perplexity=30)
        Y_sne = tsne.fit_transform(dense1_output_pca)

        fig = plt.figure(figsize = (14,12))

        # 散点图
        plt.scatter(Y_sne[:,0],Y_sne[:,1],s=2.,color=plt.cm.Set1(Y_true_label / 24.))

        # 标签图
        data = Y_sne
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        for i in range(Y_sne.shape[0]):
            plt.text(data[i,0],data[i,1], str(Y_true_label[i]),color=plt.cm.Set1(Y_true_label[i] / 24.),fontdict={'weight': 'bold', 'size': 9})
            plt.title('t-SNR at snr:{}'.format(snr))

        # plt.legend()  # 显示图示
        plt.show()

if __name__ == '__main__':
    # train(from_filename = '/media/norm_XYZ_1024_512k.hdf5',weight_file='weights/norm_res-like-512k-tap8-L6.wts.h5')
    # train(from_filename='/media/norm_XYZ_1024_128k.hdf5', weight_file='weights/norm_res-like-128k-tap8_L12.wts.h5')
    # train(from_filename='/media/norm_XYZ_1024_128k.hdf5', weight_file='weights/norm_res-like-128k.wts.h5')
    # train(from_filename='/media/XYZ_1024_512k.hdf5', weight_file='weights/res-like-1024-512k-tap8_L12.wts.h5')
    # train(from_filename='/media/XYZ_1024_1m.hdf5', weight_file='weights/res-like-1024-1m-tap8_L6.wts.h5')
    # train(from_filename='/media/XYZ_1024_1m.hdf5', weight_file='weights/vgg-like-1024-1m-tap8.wts.h5')
    # train(from_filename='/media/minmax_norm_XYZ_1024_64k.hdf5', weight_file='weights/vggbn-like-1024-temp.wts.h5')
    # train(from_filename='/media/minmax_norm_XYZ_1024_512k.hdf5', weight_file='weights/res-like-minmax-1024-512k-tap8_L6.wts.h5')
    # train(from_filename='/media/minmax-10-10_XYZ_1024_64k.hdf5', weight_file='weights/vggbn-like-1024-temp.wts.h5')
    # train(from_filename='/media/XYZ_1024_64k.hdf5', weight_file='weights/cldnn-like-1024-temp.wts.h5')
    # train(from_filename='/media/minmax-10-10_XYZ_1024_512k.hdf5', weight_file='weights/res-like-minmax-10-10-1024-512k-tap8_L6.wts.h5')
    # train(from_filename='/media/XYZ_1024_512k.hdf5', weight_file='weights/cldnn-like-1024-512k.wts.h5')
    # train(from_filename='/media/minmax-10-10_XYZ_1024_512k.hdf5', weight_file='weights/cldnn-like-1024-512k-min-10-10.wts.h5')
    # train(from_filename='/media/minmax-10-10_XYZ_1024_512k.hdf5',weight_file='weights/vgg-like-1024-512k-min-10-10.wts.h5')
    # train(from_filename='/media/GOLD_XYZ_OSC.0001_1024.hdf5',weight_file='weights/vgg-like-1024-2m.wts.h5')
    '''prepossing'''
    # test_filename = '/media/XYZ_1024_128k.hdf5'
    # f = h5py.File(test_filename, 'r')
    # X = f['X']

    # import sklearn.preprocessing
    # batch_X = X[0:2]
    #
    # for j in range(batch_X.shape[0]):
    #     print(np.max(batch_X[j], axis=0))
    #     print(np.min(batch_X[j], axis=0))
    #     print(np.var(batch_X[j], axis=0))
    #     print(np.std(batch_X[j], axis=0))
    #
    # for j in range(batch_X.shape[0]):  # each sample{ndarray[slice_len,2],ndarray[512,2]}
    #     x = batch_X[j]
    #     x = sklearn.preprocessing.scale(x)
    #     batch_X[j] = x
    #
    # import sklearn.preprocessing
    # X_scale = sklearn.preprocessing.scale(X[0],axis=0)
    #
    # for j in range(batch_X.shape[0]):
    #     print(np.max(batch_X[j], axis=0))
    #     print(np.min(batch_X[j], axis=0))
    #     print(np.var(batch_X[j], axis=0))
    #     print(np.std(batch_X[j], axis=0))

    '''predict'''
    # test_filename = '/media/XYZ_1024_128k.hdf5'
    # input_shape = (1024,2)
    # model = resnet.ResNetLikeModel(None,input_shape,classes=24)
    # # weight_file = 'weights/norm_res-like-512k.wts.h5'
    # weight_file = 'weights/res-like-1024-512k-tap8_L12.wts.h5'
    # # test_filename = '/media/norm_XYZ_1024_64k.hdf5'
    # predict(model,weight_file,test_filename)

    # '''plot t-SNR'''
    model = vggnet.VGGLikeModel(None, input_shape=[1024,2], classes=24)
    plot_tSNE(model, weight_file='weights/good/vgg-like-1024-1m-tap8.wts-58.9p.h5', test_filename='/media/XYZ_1024_128k.hdf5')

    print('test ok')
