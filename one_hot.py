import numpy as np 

def one_hot(y, n_classes):

    onehot = np.zeros((n_classes, y.shape[0]))
    for idx, val in enumerate(y.astype(int)):
        onehot[val, idx] = 1
    return onehot.T