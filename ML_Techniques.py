'''Codes for basic ML algorithms using python libraries'''

import numpy as np
import matplotlib.pyplot as plt

'''
hyperparameters = parameters that are not optimized and have to be specified by the designer of the algorithm
Learning curves - no:of iterations Vs Loss
epoch - one full pass over training set
'''
### Regression ###

# 1. Linear Regression Model
'''y = X @ w { y= label vector, X= feature matrix, w= weight vector} '''


class LinearRegression(object):
    def __init__(self):
        self.t0 = 20
        self.t1 = 100

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = X @ self.w
        return y

    def loss(self, X: np.ndarray, y: np.ndarray, reg_rate: float):
        e = y - self.predict(X)
        loss = 1/2 * (e.T@e)  # np.transpose(e) == e.T
        return loss

    def RMSE(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt((2/X.shape[0])*self.loss(X, y, 0))

    def fit(self, X: np.ndarray, y: np.ndarray, reg_rate: float):
        self.w = np.zeros((X.shape[1], y.shape[1]))
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve((reg_rate*eye + X.T@X), X.T@y)
        return self.w

    def calc_gradient(self, X: np.ndarray, y: np.ndarray, reg_rate: float) -> np.ndarray:
        return X.T@(self.predict(X)-y)

    def update_weights(self, grad: np.ndarray, lr: float) -> np.ndarray:
        return (self.w - lr*grad)

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)


def predict(X, w):
    assert X.shape[-1] == w.shape[0], "X and w are NOT shape compatible"
    y = X @ w
    return y


def loss(features, labels, weights):
    error = predict(features, weights) - labels
    loss = 1/2 * (error.T @ error)  # np.transpose(e) == e.T
    return loss

# Polynomial Regression Model


### Classification ###

# Perceptron


# KNN


# Naive Bayes


# SVM - Support Vector Machines


# Decision Trees


# Random Forest


# K-Means Clustering
