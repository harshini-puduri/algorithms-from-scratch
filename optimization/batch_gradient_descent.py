##  Gradient Descent is a first order iterative algorithm
## Idea: To take repeated steps in the opposite direction of gradient of a function at a current point.
## Because this is the direction of steepest descent.

## for 2-D --> assume L = 1/n * sum(square(y(true) - y(pred))) --> MSE
## y(pred) = wx+b, so L =  1/n * sum(square(y(true) - wx - b)), L depends on square(b)

import numpy as np
class GDRegressor:
    def __init__(self, learning_rate, epochs):
        self.w = 100 #already present
        self.b = -120 #already present
        self.lr = learning_rate
        self.epochs = epochs
    def fit(self, X, y):
        for i in range(self.epochs):
            n=2 #2D
            loss_slope_b = -2/n * np.sum(y - self.w*X.ravel() - self.b)
            loss_slope_w = -2/n * np.sum(y - self.w*X.ravel() - self.b)(X.ravel())
            self.b = self.b - (self.lr*loss_slope_b)
            self.w = self.w - (self.lr*loss_slope_w)
    def predict(self, X):
        return self.w*X + self.b
    
#for n-D
class GDRegressor:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef = None
        self.intercept = None
    def fit(self, X_train, y_train):
        ##initialise coefficients and intercept
        self.coef = np.ones(X_train.shape[0])
        self.intercept = 0
        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.coef) + self.intercept ##vectorisation
            intercept_derivative = -2 * np.mean(y_train - y_hat)
            self.intercept = self.intercept - self.lr*intercept_derivative
            coef_derivative = -2 * (np.dot(y_train - y_hat), X_train)/X_train.shape[1]
            self.coef = self.coef - self.lr*coef_derivative
    def predict(self, X_test):
        return np.dot(X_test, self.coef)+self.intercept
    
## Problem with BGD is that it is slow when dealing with large data cause in vectorisation we are loading full X_train in RAM.


