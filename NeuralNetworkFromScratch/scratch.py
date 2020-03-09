import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')
# train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
# train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

# print(train_set_x_orig.shape) # 209,64,64,3
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
# index = 25
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# print(train_set_x_flatten.shape)

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
         parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
         parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    return parameters
parameters = initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
def linear_forward_test_case():
    np.random.seed(1)
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    
    return A, W, b
def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z, cache
A, W, b = linear_forward_test_case()

# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        # Z, (A_prev,W,b)
        Z, linear_cache = linear_forward(A_prev, W, b)
        # A, (Z)
        A, activation_cache = sigmoid(Z)
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    #(A_prev,W,b), Z
    cache = (linear_cache, activation_cache)
    return A,cache

# def linear_activation_forward_test_case():
#     """
#     X = np.array([[-1.02387576, 1.12397796],
#  [-1.62328545, 0.64667545],
#  [-1.74314104, -0.59664964]])
#     W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
#     b = 5
#     """
#     np.random.seed(2)
#     A_prev = np.random.randn(3,2)
#     W = np.random.randn(1,3)
#     b = np.random.randn(1,1)
#     return A_prev, W, b
# A_prev, W, b = linear_activation_forward_test_case()

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return X, parameters

def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X
    for l in range(1,L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
print(caches)

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    
    cost = np.squeeze(cost)      
    return cost
