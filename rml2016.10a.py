# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses

import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)
import numpy as np
import matplotlib.pyplot as plt
import pickle, random, sys
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler

import mltools,rmldataset2016
import rmlmodels.CNN2Model as cnn2

(mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx) = \
    rmldataset2016.load_data(filename ="data/RML2016.10a_dict.pkl", train_rate = 0.5)

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
print(classes)

# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,2,128,1] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 1024  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
model = cnn2.CNN2Model(None, input_shape=in_shp,classes=len(classes))

rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=rmsp)


def scheduler(epoch):
    if int(epoch % 80) == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {} at epoch{}".format(lr * 0.5, epoch))
        return K.get_value(model.optimizer.lr)
    else:
        print("epoch({}) lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)

filepath = 'weights/CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks = [reduce_lr,
                 keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                # ,keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
                ]
                    )
# history = model.fit(X_train,
#     Y_train,
#     batch_size=batch_size,
#     epochs=nb_epoch,
#     verbose=1,
#     validation_data=(X_test, Y_test),
#     callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
#                     )

# we re-load the best weights once training is finished
# model.load_weights(filepath)
mltools.show_history(history)

# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

def plot_tSNE(model,filename="data/RML2016.10a_dict.pkl"):
    from keras.models import Model
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = \
        rmldataset2016.load_data(filename, train_rate=0.80)

    #设计中间层输出的模型
    dense2_model = Model(inputs=model.input,outputs=model.get_layer('dense1').output)

    #提取snr下的数据进行测试
    for snr in [s for s in snrs if s > 14]:
        test_SNRs = [lbl[x][1] for x in test_idx]       #lbl: list(mod,snr)
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        #计算中间层输出
        dense2_output = dense2_model.predict(test_X_i,batch_size=32)
        Y_true = np.argmax(test_Y_i,axis=1)

        #PCA降维到50以内
        pca = PCA(n_components=50)
        dense2_output_pca = pca.fit_transform(dense2_output)

        #t-SNE降为2
        tsne = TSNE(n_components=2,perplexity=5)
        Y_sne = tsne.fit_transform(dense2_output_pca)

        fig = plt.figure(figsize = (14,12))

        # 散点图
        # plt.scatter(Y_sne[:,0],Y_sne[:,1],s=5.,color=plt.cm.Set1(Y_true / 11.))

        # 标签图
        data = Y_sne
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        for i in range(Y_sne.shape[0]):
            plt.text(data[i,0],data[i,1], str(Y_true[i]),color=plt.cm.Set1(Y_true[i] / 11.),fontdict={'weight': 'bold', 'size': 9})
            plt.title('t-SNR at snr:{}'.format(snr))

        # plt.legend()  # 显示图示
        fig.show()

def predict(model,filename="data/RML2016.10a_dict.pkl"):
    (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = \
        rmldataset2016.load_data(filename, train_rate=0.5)
    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm,_,_ = mltools.calculate_confusion_matrix(Y_test,test_Y_hat,classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes)

    # Plot confusion matrix
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(test_Y_i,test_Y_i_hat,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)

        mltools.plot_confusion_matrix(confnorm_i, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)(ACC=%2f)" % (snr,100.0*acc[snr]))


    # Save results to a pickle file for plotting later
    print(acc)
    fd = open('predictresult/cnn2_d0.5.dat','wb')
    pickle.dump( ("CNN2", 0.5, acc) , fd )

    # Plot accuracy curve
    plt.plot(snrs, [acc[i] for i in snrs])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
    plt.show()

if __name__ == '__main__':
    plot_tSNE(model)

    predict(model, filename="data/RML2016.10a_dict.pkl")
