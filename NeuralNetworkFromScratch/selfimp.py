import numpy as np 
np.random.seed(1)

class NeuralNetWork():
    
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
            d = obj.gradient(dZ, caches[i-1])
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

obj = NeuralNetWork([2,2,1])
#print('shape',np.array([[1,2],[3,4] ]).shape)
#print('test sig', obj.sigmoid(np.array([[1,2],[3,4] ])))

X= np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

obj.fit(X,Y, 5000, 1.1)

lable, _ = obj.predict(np.array([[0,1]]))
print(lable)
