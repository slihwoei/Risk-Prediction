#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [tuple(self.y_train[i]) for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def get_params(self, deep=True):
        return {'k': self.k}

    def set_params(self, **params):
        self.k = params['k']
        return self

