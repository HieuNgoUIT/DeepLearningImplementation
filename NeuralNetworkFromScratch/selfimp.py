import numpy as np 
np.random.seed(42)

class NeuralNetWork():
    layer = {}
    def create_matrix_base_on_layer(self, previous_layer, current_layer):
        return np.random.rand(previous_layer, current_layer)

    def create_structure(self, number_each_layer):
        l = len(number_each_layer)
        for i in range(1,l):
            self.layer[i]  = self.create_matrix_base_on_layer(number_each_layer[i-1], number_each_layer[i])
        return self.layer

    def calculate_Z(self, X, weights):
        #print("X shapre",X.shape)
        #print('w shape', weights.shape)
        return np.dot(X, weights)

    def sigmoid(self, Z):
        return 1/ (1 + np.exp(-Z))

    def foward_prop(self, X, weights):
        l = len(weights)
        A = X
        print(X.shape)
        for i in range(1,l+1):
            X = self.calculate_Z(A, weights[i])
            A = self.sigmoid(X)
            X = A
        return A
    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1- AL)))
        return cost
            

obj = NeuralNetWork()
weights = obj.create_structure([3,2,1])

X= np.array([[1,2,3], [4,5,6]])
Y = np.array([[1]])

AL = obj.foward_prop(X,weights)

print(obj.compute_cost(AL,Y))