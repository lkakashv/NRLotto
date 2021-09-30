import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Lottonumbers = 49.


data = pd.read_csv('germanlotto.csv')

data = np.array(data)
m, n = data.shape
print(m,n)

# Arranging Data, Y_train is answers array, X_train is training data to train neural network, we need Y_train for backpropagation to calculate derivatives (in easy way how correct/incorect was a neural network and twik/update weigths and bias based on it)
data_train = data[:m].T # we Transpose numbers so winning numbers now are in colums instead of rows
Y_train = np.arange(1)
Y_train[:] = 1
X_train = data_train[:n]
X_train = X_train / Lottonumbers # we devide to Lottonumbers becouse numbers in ressults are from 1 to Lottonumbers, so our numbers will stay in 0 - 1 range
_,m_train = X_train.shape




def init_params():
    W1 = np.random.rand(10, n) - 0.5 # - 0.5 is used to have numbers between -0.5 and 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(1, 10) - 0.5
    b2 = np.random.rand(1, 1) - 0.5

    return W1, b1, W2, b2

# ReLU function description https://bit.ly/2ZKN6yv
def ReLU(Z): 

    return np.maximum(Z, 0) # basicly in our situtation creates array where all the element are from 0 and up till 1


# softmax Function description https://www.bit.ly/3lRkGKJ
def softmax(Z):
 
    A = np.exp(Z) / sum(np.exp(Z)) 
    
    return A

# Input nodes = n, n is how many winning numbers are in Lotto, hidden layer nodes 10, output layer nodes 1
# Z0 is firs layer(Input Layer), Z1 is second Layer and Z2 is last layer (Output Layer)
# W1 is weights for First Layer and b1 is bias for first layers
# first step we need to multiply all weigts to input arguments for us it's 20 input nodes and add bias for hidden layer nodes (10 hidden nodes)
# A is a activation function, A1 is ReLu and A2 is SofMax (Soft max is between 0-1), A1 is used on hidden nodes and A2 on output nodes  
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # all weigths-s (W1) are multiplyed to all X (X_train)-z and add bias for hidden 10 nodes
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    
    return Z1, A1, Z2, A2


# Calculate ReLU derivative 
def ReLU_deriv(Z):
    return Z > 0

# Backpropagation function to update weights based on error rate calculated with derivatives
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    dZ2 = A2 - 1 #as we have only one output node we know is it's correct than it's 1, so to calculate derivative we need to substract 1 from our ressult
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# Update/Twick Weigths and bias
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    
 
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):

    return np.sum(predictions == Y) / Y.size # counts how many correct answers we done and multiplys on total correct answers to get percentage

# MAIN function that uses gradient decent to train neural network

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            
            predictions = get_predictions(A2)
            
            print("Accuracy: ",get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)