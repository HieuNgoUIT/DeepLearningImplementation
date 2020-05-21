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
            self.b[i] = np.zeros((1,number_each_layer[i]))
        return self.b

    def calculate_Z(self, X, weights, b):
        #print("X shapre",X.shape)
        #print('w shape', weights.shape)
        paramater = {}
        paramater['A_prev'] = X
        paramater['w'] = weights
        #paramater['A_prev'] = X
        #print("wT",weights.T.shape)
        #print('basfd',b.shape)
        return np.dot(X, weights)+ b  , paramater

    def sigmoid(self, Z):
        return 1/ (1 + np.exp(-Z))

    def foward_prop(self, X, weights, b):
        l = len(weights)
        A = X
        caches = []
        for i in range(1,l+1):
            Z, cache = self.calculate_Z(A, weights[i], b[i])
            caches.append(cache)
            A = self.sigmoid(Z)
        return A, caches
    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1- AL)))
        return cost
    
    def gradient(self, dZ, paramater):
        d = {}
        #print('a',paramater['A_prev'].shape)
        #print('w',paramater['w'].shape)
        d['w'] = np.dot(paramater['A_prev'].T, dZ)
        d['b'] = dZ
        d['a'] = np.dot(dZ, paramater['w'].T)
        return d
    def update_param(self, paramater):
        l = len(number_each_layer)
        for i in range(l+1,-1,-1):
            d = obj.gradient(dZ, caches[i-1])
            weights[i] -= d['w']
            b[i] -= d['b']
            dZ = d['a_prev']

obj = NeuralNetWork()

weights = obj.create_structure([3,2])
#print('weights',weights)
print('w',weights[1].shape)
b = obj.create_b_base_on_layer([3,2])
#print('b', b)
print('b',b[1].shape)

X= np.array([[1,1,1]])
print('a',X.shape)
print("______________________")
Y = np.array([[3,2]])

AL, caches = obj.foward_prop(X,weights,b)
#print(caches[1])


dZ = AL -Y
#print(dZ)
#print('dzshape',dZ.shape)
d = obj.gradient(dZ, caches[0])
print('w shape',d['w'].shape)
print('b shape',d['b'].shape)
print('a shape',d['a'].shape)

# #print(AL)
# #print(obj.compute_cost(AL,Y))

# #print(np.dot([1,2,3], 0.5))
