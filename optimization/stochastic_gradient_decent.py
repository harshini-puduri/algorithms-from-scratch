## SGD is designed to optimise linear regression models
## it supports L1,L2, ElasticNet regularisations
##it works better when features are on a similar sclae, hence we use StandardScaler() to standardise the features
import numpy as np
class SGDRegressor:
    def __init(self,learning_rate,epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.intercept = None
        self.coef = None
    def fit(self, X_train, y_train):
        #initialise coef and intercept
        self.coef = np.ones(X_train.shape[1])
        self.intercept = 0
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0,X_train.shape[0])
                y_hat = np.dot(X_train[idx],self.coef) + self.intercept
                intercept_derivative = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept = self.intercept - self.lr * intercept_derivative
                coef_derivative = -2 * np.dot(y_train[idx] - y_hat, X_train[idx])/X_train.shape[0]
                self.coef = self.coef - self.lr * coef_derivative
    def predict(self, X_test):
        return np.dot(X_test, self.coef) + self.intercept
    

