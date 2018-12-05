import tensorflow as tf
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


class tensorflow_pca():
    def __init__(self,dtype,data):
        self.dtype = dtype
        self.data = data

    def fit(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(self.dtype, shape=self.data.shape)
            # Perform SVD
            singular_values, u, _ = tf.svd(self.X)
            # Create sigma matrix
            sigma = tf.diag(singular_values)

        with tf.Session(graph=self.graph) as session:
            self.u, self.singular_values, self.sigma = session.run([u, singular_values, sigma],
                                                                   feed_dict={self.X: self.data})

    def reduce(self, n_dimensions=None, keep_info=None):
        if keep_info:
            normalized_singular_values = self.singular_values / sum(self.singular_values)
            # Create the aggregated ladder of kept information per dimension
            ladder = np.cumsum(normalized_singular_values)
            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1

            n_dimensions = index

        with self.graph.as_default():
            # Cut out the relevant part from sigma
            sigma = tf.slice(self.sigma, [0, 0], [self.data.shape[1], n_dimensions])
            # PCA
            pca = tf.matmul(self.u, sigma)
        with tf.Session(graph=self.graph) as session:
            return session.run(pca, feed_dict={self.X: self.data})

if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = tensorflow_pca(dtype = tf.float32,data=X)

    pca.fit()

    X_r = pca.reduce(n_dimensions=2)

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


