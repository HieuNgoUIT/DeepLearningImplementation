import numpy as np 
np.random.seed(1)

class NeuralNetWork():
    weights = {}
    b = {}
    def create_matrix_base_on_layer(self, previous_layer, current_layer):
        return np.random.rand(previous_layer, current_layer) * 0.01

    def create_structure(self, number_each_layer):
        l = len(number_each_layer)
        for i in range(1,l):
            self.weights[i]  = self.create_matrix_base_on_layer(number_each_layer[i-1], number_each_layer[i])
        return self.weights

    def create_b_base_on_layer(self, number_each_layer):
        l = len(number_each_layer)
        for i in range(1,l):
            self.b[i] = np.zeros((number_each_layer[i],1))
        return self.b

    def calculate_Z(self, X, weights, b):
        #print("X shapre",X.shape)
        #print('w shape', weights.shape)
        paramater = {}
        paramater['A_prev'] = X
        paramater['w'] = weights
        #paramater['A_prev'] = X
        #print("wT",weights.T.shape)
        return np.dot(X, weights) #+ b  

    def sigmoid(self, Z):
        return 1/ (1 + np.exp(-Z))

    def foward_prop(self, X, weights, b):
        l = len(weights)
        A = X
        caches = []
        #print(X.shape)
        for i in range(1,l+1):
            #print(i)
            X, cache = self.calculate_Z(A, weights[i], b[i])
            #print('X' , X)
            caches.append(cache)
            A = self.sigmoid(X)
            #print('A', A)
            X = A
        return A, caches
    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1- AL)))
        return cost
    
    def gradient(self, dZ, paramater):
        d = {}
        d['w'] = paramater['A_prev'] * dZ
        d['b'] = dZ
        d['a'] = paramater['w'] * dZ
        return d
    def update_param(self, paramater):
        self.weights

obj = NeuralNetWork()

weights = obj.create_structure([3,2,1])
print('weights',weights)
print(weights[1].shape)
b = obj.create_b_base_on_layer([3,2,1])
print('b', b)
print(b[1].shape)



X= np.array([[1,1,1] ])
#print(X.shape)
print("Z",obj.calculate_Z(X,weights[1],b[1]))

# Y = np.array([[1]])
# #print(obj.sigmoid(2.88))

# AL, caches = obj.foward_prop(X,weights,b)
# #print("caches",caches)
# dZ = AL -Y
# #print('dz', dZ)
# #
# d = obj.gradient(dZ, caches[0])
# #print(d)

# #print(AL)
# #print(obj.compute_cost(AL,Y))

# #print(np.dot([1,2,3], 0.5))
