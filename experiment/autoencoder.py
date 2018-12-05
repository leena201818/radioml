import tensorflow as tf
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


class autoencoder():
    def __init__(self,dtype,features):
        self.dtype = dtype
        self.features = features
        self.encoder = {}

    def batch_generator(self,features,batch_size = 10):
        num = int(np.ceil(features.shape[0]/batch_size))
        for i in range(num):
            beg = i*batch_size
            end = np.min([(i+1)*batch_size,features.shape[0]])
            yield features[beg:end,...]

    def fit(self,n_dimensions,epochs=10,batch_size=10):
        graph = tf.Graph()
        with graph.as_default():
            #输入
            X = tf.placeholder(self.dtype,shape=(None,self.features.shape[1]))
            #网络
            encoder_weights = tf.Variable(tf.random_normal(shape=(self.features.shape[1],n_dimensions)))
            encoder_bias = tf.Variable(tf.zeros(shape=[n_dimensions]))

            decoder_weights = tf.Variable(tf.random_normal(shape=[n_dimensions,self.features.shape[1]]))
            decoder_bias = tf.Variable(tf.zeros(shape=[self.features.shape[1]]))

            encoding = tf.nn.sigmoid(tf.add(tf.matmul(X,encoder_weights),encoder_bias))

            predicted_x = tf.nn.sigmoid(tf.add(tf.matmul(encoding,decoder_weights),decoder_bias))

            cost = tf.reduce_sum(tf.pow(tf.subtract(predicted_x,X),2))

            cost_weights = tf.nn.l2_normalize(encoder_weights)
            cost_weights2 = tf.nn.l2_normalize(decoder_weights)

            cost_total = cost

            optimizer = tf.train.AdamOptimizer().minimize(cost_total)

        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            for i in range(epochs):
                print('epoch {} start'.format(i))
                for batch_x in self.batch_generator(self.features):
                    self.encoder['weights'],self.encoder['bias'],_ = session.run([encoder_weights,encoder_bias,optimizer],feed_dict={X:batch_x})
                print('epoch {} end'.format(i))

    def reduce(self):
        return np.add(np.matmul(self.features,self.encoder['weights']),self.encoder['bias'])

if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    ae = autoencoder(dtype = tf.float32,features = X)

    ae.fit(n_dimensions=2,epochs=1000,batch_size=50)

    X_r = ae.reduce()

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    plt.title('auto encoder of IRIS dataset')
    plt.show()
    print('reduce ok')


