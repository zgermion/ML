from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)
    d = {}
    
    for k in k_list:
        d[k] = np.array([])
        

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))
        
    algorithm = ''
    metric = ''
    weights = ''
    batch_size = 0
    
        
    if "algorithm" in kwargs:
        algorithm = kwargs["algorithm"]
    else:
        algorithm = 'my_own'
        
    if "metric" in kwargs:
        metric = kwargs["metric"]
    else:
        metric = 'euclidean'
        
    if "weights" in kwargs:
        weights = kwargs["weights"]
    else:
        weights = 'uniform'
        
    if "batch_size" in kwargs:
        batch_size = kwargs["batch_size"]
    else:
        batch_size = X.shape[0]
        
    k_max = max(k_list)
    
    s = BatchedKNNClassifier(k_max, algorithm=algorithm, metric=metric, weights=weights, batch_size=batch_size)
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s.fit(X_train, y_train)
        f = s.kneighbors(X_test, return_distance = True)
        distances = f[0]
        indixes = f[1]
        for k in k_list:
            y_pred = s._predict_precomputed(indixes[:, 0:k], distances[:, 0:k])
            d[k] = np.append(d[k], accuracy_score(y_test, y_pred))  
    return d

