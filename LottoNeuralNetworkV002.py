import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


Lottonumbers = 49.


data = pd.read_csv('germanlotto_train.csv')


data = np.array(data) #convert pandas array into numpy array
m, n = data.shape #m = rows, n = colums in 2d array
np.random.shuffle(data) # shuffle before splitting into dev and training sets
print(m,n)

# Arranging Data, Y_train is answers array, X_train is training data to train neural network, we need Y_train for backpropagation to calculate derivatives (in easy way how correct/incorect was a neural network and twik/update weigths and bias based on it)
data_train = data[:m].T # we Transpose numbers so winning numbers now are in colums instead of rows
Y_train = data_train[0] #take firs row (all the correct labels, 1 - is winning numbers, 0 - not winning numbers)
X_train = data_train[1:n] #take all the remining rows (in our situatuin 7)
X_train = X_train / Lottonumbers # we devide to Lottonumbers becouse numbers in ressults are from 1 to Lottonumbers, so our numbers will stay in 0 - 1 range
_,m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, n-1) - 0.5 # - 0.5 is used to have numbers between -0.5 and 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(2, 10) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0) # basicly in our situtation creates array where all the element are from 0 and up

def softmax(Z):
 
    A = np.exp(Z) / sum(np.exp(Z)) # https://www.bit.ly/3lRkGKJ
    return A

# Input nodes 7, hidden nlayer nodes 10, output layer nodes 2
# Z0 is firs layer(Input Layer), Z1 is second Layer and Z2 is last layer (Output Layer)
# W1 is weights for First Layer and b1 is bias for first layers
# first step we need to multiply all weigts to input arguments for us it's 7 input nodes and add bias for hidden layer nodes (10 hidden nodes)
# A is a activation function, A1 is ReLu and A2 is SofMax (Soft max is between 0-1), A1 is used on hidden nodes and A2 on output nodes  
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # all weigths-s (W1) are multiplyed to all X (X_train)-z so we need 7 Weigths (W1) to multiply to 7 input nodes (X_train) and add bias for hidden 10 nodes
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
 
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

# Used to calculate derivative for 2 output nodes (A2)
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) #creates and fills with zeros one_hot_Y 2D array size of our train data array (Y_train) and maximum number in Y_train array + 1 (in our case will be 2)
    one_hot_Y[np.arange(Y.size), Y] = 1 # in our 2D array puts 1 in the column where should be a rigth answer, if numbers are right than it will be 1 if numbers are wrong than it will be 0
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2



def get_predictions(A2):

    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print ("Sum: ",np.sum(predictions == Y))
    #print ("Y Size",Y.size)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #print("Size Weigths1",W1.size,"----->>>Weights W1",W1)
        if i % 10 == 0:
            print("Iteration: ", i)
            
            predictions = get_predictions(A2)
            
            print("Accuracy:",get_accuracy(predictions, Y))
    return W1, b1, W2, b2



W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

