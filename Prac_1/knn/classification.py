import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        if self._weights == 'uniform':
            A = self._labels[indices]
            N = A.max()+1
            id = A + (N*np.arange(A.shape[0]))[:,None]
            labels_new = [np.argmax(np.bincount(id.ravel(),minlength=N*A.shape[0]).reshape(-1,N), axis = 1)]
            return labels_new[0]
        else:
            k = []
            for i in range(self._labels[indices].shape[0]):
                w = 1/(distances[i] + self.EPS)
                sum_ans = 0
                l = 0
                for j in set(self._labels[indices[i]]):
                    sum_1 = sum(w[self._labels[indices[i]] == j])
                    if sum_1 > sum_ans:
                        sum_ans = sum_1
                        l = j
                k = k + [l]
            return np.array(k)
                

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        else:
            nu = super().kneighbors(X[self._batch_size*0 : self._batch_size*1], return_distance=return_distance)
            L1 = [nu[0]]
            L2 = [nu[1]]
            arr1 = np.ndarray((X.shape[0], nu[1].shape[1]))
            arr2 = np.ndarray((X.shape[0], nu[1].shape[1]))
            
            k = int(X.shape[0] / self._batch_size)
            for i in range(1, k):
                nu = super().kneighbors(X[self._batch_size*i : self._batch_size*(i+1)], return_distance=return_distance)
                L1 = L1+[nu[0]]
                L2 = L2+[nu[1]]
            nu = super().kneighbors(X[self._batch_size*k : self._batch_size*k+X.shape[0] % self._batch_size], return_distance=return_distance)
            
            L1 = L1+[nu[0]]
            L2 = L2+[nu[1]]
            
            arr1 = np.vstack(L1)
            arr2 = np.vstack(L2)
            
            if return_distance:
                return arr1, arr2
            else:
                return arr2