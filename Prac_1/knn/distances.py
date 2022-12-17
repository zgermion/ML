import numpy as np


def euclidean_distance(x, y):
    n = x.shape[0]
    m = y.shape[0]

    A = (x * x).sum(axis=1).reshape((n, 1)) * np.ones(shape=(1, m))
    B = (y * y).sum(axis=1) * np.ones(shape=(n, 1))
    res = A + B - 2 * x.dot(y.T)

    return np.sqrt(res)


def cosine_distance(x, y):
    one_x = np.linalg.norm(x, axis=1)
    one_y = np.linalg.norm(y, axis=1)
    n = np.shape(one_x)[0]
    one_x = one_x.reshape((n, 1))
    m = np.shape(one_y)[0]
    one_y = one_y.reshape((1, m))

    return 1-x.dot(y.T) / (one_x * one_y)