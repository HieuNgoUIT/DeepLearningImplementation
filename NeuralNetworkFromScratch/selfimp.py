import numpy as np 
np.random.seed(42)

class NeuralNetWork():
    layer = {}
    b = {}
    def create_matrix_base_on_layer(self, previous_layer, current_layer):
        return np.random.rand(previous_layer, current_layer)

    def create_structure(self, number_each_layer):
        l = len(number_each_layer)
        for i in range(1,l):
            self.layer[i]  = self.create_matrix_base_on_layer(number_each_layer[i-1], number_each_layer[i])
        return self.layer

    def calculate_Z(self, X, weights, b):
        #print("X shapre",X.shape)
        #print('w shape', weights.shape)
        return np.dot(X, weights) + b

    def create_b_base_on_layer(self, layers):
        l = len(layers)
        for i in range(1,l):
            self.b[i] = np.random.rand()
        return self.b

    def sigmoid(self, Z):
        return 1/ (1 + np.exp(-Z))

    def foward_prop(self, X, weights, b):
        l = len(weights)
        A = X
        #print(X.shape)
        for i in range(1,l+1):
            X = self.calculate_Z(A, weights[i], b[i])
            #print('X' , X)
            A = self.sigmoid(X)
            #print('A', A)
            X = A
        return A
    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1- AL)))
        return cost
    
    def gradient(self, dZ):
        dw = a_prev * dZ
        db = dZ
        da = w * dZ

obj = NeuralNetWork()

weights = obj.create_structure([3,2,1])
#print('weights',weights)
b = obj.create_b_base_on_layer([3,2,1])
#print('b', b)




X= np.array([[1,2,3] ])
Y = np.array([[1]])
#print(obj.sigmoid(2.88))

AL = obj.foward_prop(X,weights,b)
print(AL)
#print(obj.compute_cost(AL,Y))