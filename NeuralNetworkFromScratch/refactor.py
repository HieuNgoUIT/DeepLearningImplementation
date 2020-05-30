import numpy as np 
np.random.seed(1)

class NeuralNetwork():
    def __init__(self, layer_dims):
        self.parameters = {}
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims) -1 

    def init_parameters(self):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.rand(self.layer_dims[l -1], self.layer_dims[l])
            self.parameters['b' + str(l)] = np.zeros((1, self.layer_dims[l]))
            
            assert(self.parameters['W' + str(l)].shape == (self.layer_dims[l -1], self.layer_dims[l]))
            assert(self.parameters['b' + str(l)].shape == (1, self.layer_dims[l]))

    def linear_foward(self, A, w, b):
        Z = A.dot(w) + b
        assert(Z.shape == (A.shape[0], w.shape[1]))
        cache = (A, w, b)
        return Z, cache
    
    def sigmoid(self, Z):
        A = 1/ (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self ,Z):
        A = np.maximum(0,Z)
        cache = Z
        return A, cache

    def linear_activation_foward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_foward(A_prev,W, b)
            A , activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            Z, linear_cache = self.linear_foward(A_prev,W, b)
            A , activation_cache = self.relu(Z)
        
        assert(A.shape == (A_prev.shape[0], W.shape[1]))
        cache = (linear_cache, activation_cache)
        
        return A, cache

    def L_model_foward(self,X):
        caches = []
        A = X
        for l in range(1, self.L):
            A_prev = A
            A, cache = self.linear_activation_foward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation="sigmoid")
            caches.append(cache)
        AL, cache = self.linear_activation_foward(A, self.parameters['W' + str(self.L)], self.parameters['b' + str(self.L)], activation="relu")
        caches.append(cache)

        #assert(AL.shape == ())
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1- AL)))
        return cost
    
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1./m * np.dot(A_prev.T, dZ)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(dZ, W.T)

        #assert (dA_prev.shape == A_prev.shape)
        #assert (dW.shape == W.shape)
        #assert (db.shape == b.shape)

        return dA_prev, dW, db

    def relu_backward(self,dA, cache):
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
        
        #assert (dZ.shape == Z.shape)
        
        return dZ

    def sigmoid_backward(self, dA, cache):
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
        
        #assert (dZ.shape == Z.shape)
        
        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def gradient(self, dZ, caches):
        d = {}
        m = caches['A_prev'].shape[0]
        d['w'] = (1/m) * np.dot(caches['A_prev'].T, dZ)
        d['b'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        d['a'] = np.dot(dZ, caches['w'].T) 
        return d

    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
            
        return parameters

    def fit(self, X, Y, epoch, alpha):
        for _ in range(epoch):
            AL, caches = self.foward_prop(X, self.weights, self.b)
            cost = self.compute_cost(AL, Y)
            print(cost)
            dZ = AL - Y
            self.update_param(caches, dZ, alpha)
            
    def predict(self, X):
        return self.foward_prop(X, self.weights,self.b)




obj = NeuralNetwork([3,256,1])
#print('shape',np.array([[1,2],[3,4] ]).shape)
#print('test sig', obj.sigmoid(np.array([[1,2],[3,4] ])))

X= np.array([[0,1,1], [0,1,0], [1,0,1], [0,0,1], [1,1,1]])
Y = np.array([[0], [0], [0], [0] , [1]])

obj.fit(X,Y, 100, 0.1)

lable, _ = obj.predict(np.array([[1,1,1]]))
print(lable)


# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import scipy
# from PIL import Image
# from scipy import ndimage
# from lr_utils import load_data
# import scipy.misc

# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()


# index = 7
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# m_train = train_set_x_orig.shape[0]
# m_test = test_set_x_orig.shape[0]
# num_px = train_set_x_orig.shape[1] 


# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))


# # Reshape the training and test examples


# train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
# test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)
# train_set_y = train_set_y.T
# test_set_y= test_set_y.T

# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

# my_neural = NeuralNetwork([12288, 20, 7, 5, 1])
# my_neural.fit(train_set_x_flatten, train_set_y, 1000, 1)