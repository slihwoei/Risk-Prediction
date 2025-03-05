#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
    
class Perceptron:
    
    def __init__(self, num_features,learning_rate=0.0001,max_iter=1000):
        self.num_features = num_features
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights_d = np.zeros(num_features)
        self.bias = 0
    
    def predict(self, x):
        raw_predictions = np.dot(x, self.weights_d) + self.bias
        predictions = np.where(raw_predictions > 0, 1, -1)
        return predictions
    
    def fit(self, X, y):
        for i in range(self.max_iter):
            for j in range(len(X)):
                prediction = self.predict(X[j])
                if prediction * y[j] <= 0:
                    update = self.learning_rate * (y[j] - prediction)
                    self.weights_d += update * X[j]
                    self.bias += update
    
    def get_params(self, deep=True):
        return {
            'num_features': self.num_features,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter
        }
    
    def set_params(self, **params):
        if 'num_features' in params:
            self.num_features = params['num_features']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
        return self

