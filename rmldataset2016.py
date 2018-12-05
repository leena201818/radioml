import pickle
import numpy as np

def load_data(filename = "data/RML2016.10a_dict.pkl",train_rate = 0.5):

    Xd = pickle.load(open(filename,'rb'),encoding='iso-8859-1')
    # snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    # same with the following without using map function,but 'for' method
    # mods = set([ k[0] for k in Xd.keys()])
    # mods = list(mods)
    # mods = sorted(mods)
    #
    # snrs = sorted(list(set([k[1] for k in Xd.keys()])))

    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]

    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     #ndarray(1000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)                    #(220000,2,128)  mods * snr * 1000,total 220000 samples

    # Partition the data
    # into training and test sets of the form we can train/test on
    # while keeping SNR and Mod labels handy for each
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * train_rate)

    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))

    X_train = X[train_idx]
    X_test =  X[test_idx]

    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    # yy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    # Y_one_hot = np.zeros([len(lbl),len(mods)])
    # for i in range(len(lbl)):
    #     Y_one_hot[i,mods.index(lbl[i][0])] = 1.
    #
    # Y_train2 = Y_one_hot[train_idx]
    # Y_test2 = Y_one_hot[test_idx]
    #
    # print( np.all(Y_test2 == Y_test) )

    return (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = load_data()
