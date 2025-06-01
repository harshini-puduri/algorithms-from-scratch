import numpy as np
def perceptron(X,y):
    learning_rate = 0.01
    epochs = 1000 #generally
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    for i in range(epochs):
        j = np.random.randint(0,100) #if we have 100 samples
        y_hat = step(np.dot(X[j], weights))
        weights = weights + (learning_rate*(y[j]-y_hat)*X[j])
    return weights[0], weights[1:]

def step(a):
    return 1 if a>0 else 0

##using loss function
def perceptron(X,y):
    w1=w2=b=1
    lr = 0.1
    epochs = 1000
    for i in range(epochs):
        for j in range(X.shape[0]):
            z = w1*X[j][0] + w2*X[j][1] + b
            if z*y[j] < 0:
                w1 = w1 + lr*y[j]*X[j][0]
                w2 = w2 + lr*y[j]*X[j][1]
                b = b + lr*y[j]
    return w1, w2, b