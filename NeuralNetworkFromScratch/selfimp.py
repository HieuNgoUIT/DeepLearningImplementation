import numpy as np 
np.random.seed(1)

class NeuralNetwork():
    
    def __init__(self, number_each_layer):
        self.weights = {}
        self.b = {}
        self.number_each_layer = number_each_layer
        self.l = len(self.number_each_layer)
        self.create_weights()
        self.create_b()

    def create_matrix_base_on_layer(self, previous_layer, current_layer):
        return np.random.rand(previous_layer, current_layer) * 0.01

    def create_weights(self):
        for i in range(1, self.l):
            self.weights[i]  = self.create_matrix_base_on_layer(self.number_each_layer[i-1], self.number_each_layer[i])

    def create_b(self):
        for i in range(1,self.l):
            self.b[i] = np.zeros((1, self.number_each_layer[i]))

    def calculate_Z(self, X, weights, b):
        paramater = {}
        paramater['A_prev'] = X
        paramater['w'] = weights
        paramater['b'] = b
        return np.dot(X, weights)+ b  , paramater

    def sigmoid(self, Z):
        return 1/ (1 + np.exp(-Z))

    def foward_prop(self, X, weights, b):
        A = X
        caches = []
        for i in range(1, self.l):
            Z, cache = self.calculate_Z(A, weights[i], b[i])
            caches.append(cache)
            A = self.sigmoid(Z)
        return A, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1- AL)))
        return cost
    
    def gradient(self, dZ, caches):
        d = {}
        m = caches['A_prev'].shape[0]
        #print('a',paramater['A_prev'].shape)
        #print('w',paramater['w'].shape)
        d['w'] = (1/m) * np.dot(caches['A_prev'].T, dZ)
        d['b'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        d['a'] = np.dot(dZ, caches['w'].T) 
       # print('dashape', d['a'].shape)
        return d

    def update_param(self, caches, dZ, alpha):
        for i in range(self.l-1, 0, -1):
            d = self.gradient(dZ, caches[i-1])
            self.weights[i] -= alpha *  d['w']
            self.b[i] -= alpha * d['b']
            #print('a',caches[i-1]['A_prev'])
            #print('w',caches[i-1]['w'])
            #print('b',caches[i-1]['b'])
            Z , _ = self.calculate_Z(caches[i-2]['A_prev'], caches[i-2]['w'], caches[i-2]['b'])
            s = self.sigmoid(Z)
            s1 = 1 -  self.sigmoid(Z)
            c = s*s1
            #print('c shape',c.shape)
            dZ =  c * d['a'] 
            #print(dZ.shape)
            #print('c',dZ.shape)
    def fit(self, X, Y, epoch, alpha):
        for _ in range(epoch):
            AL, caches = self.foward_prop(X, self.weights, self.b)
            cost = self.compute_cost(AL, Y)
            print(cost)
            dZ = AL - Y
            self.update_param(caches, dZ, alpha)
    def predict(self, X):
        return self.foward_prop(X, self.weights,self.b)

# obj = NeuralNetwork([2,1])
# #print('shape',np.array([[1,2],[3,4] ]).shape)
# #print('test sig', obj.sigmoid(np.array([[1,2],[3,4] ])))

# X= np.array([[0,1], [1,0], [1,1], [0,0]])
# Y = np.array([[1], [1], [1], [0]])

# obj.fit(X,Y, 5000, 0.1)

# lable, _ = obj.predict(np.array([[1,1]]))
# print(lable)


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import scipy.misc

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


index = 7
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1] 


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# Reshape the training and test examples


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)
train_set_y = train_set_y.T
test_set_y= test_set_y.T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

my_neural = NeuralNetwork([12288, 20, 7, 5, 1])
my_neural.fit(train_set_x_flatten, train_set_y, 5000, 0.1)