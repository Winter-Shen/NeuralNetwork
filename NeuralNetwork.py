import numpy as np
from tensorflow.keras.losses import MeanSquaredError

class InputLayer:
    def __init__(self, dim):
        self.dim = dim
        self.livelyNode = list(range(dim))
    def activeNode(self, nodeIndex):
        self.livelyNode = nodeIndex
        return self
    def input(self, data):
        if(data.shape[1] != self.dim):
            return None
        else:
            self.X = data
            return self
    def forwardPropogation(self):
        self.Y = self.X[:,self.livelyNode]
        return self
    def output(self):
        return self.Y
class Layer:
    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim
        self.livelyNode = list(range(outDim))
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.inDim, self.outDim)).numpy())
    def setWeight(self, weight):
        if(weight.shape[0] != self.inDim or weight.shape[1]!= self.outDim):
            return None
        else:
            self.weight = weight
            return self
    def activeNode(self, nodeIndex):
        self.livelyNode = nodeIndex
        return self
    def input(self, data):
        if(data.shape[1] != self.inDim):
            return None
        else:
            self.X = data
            self.size = data.shape[0]
            return self
    def forwardPropogation(self):
        self.Y = self.X.dot(self.weight[:,self.livelyNode])
        return self
    def output(self):
        return self.Y
    def setDy(self, dy):
        if(dy.shape[0] != self.size or dy.shape[1]!= self.outDim):
            return None
        else:
            self.dy = dy
            return self
    def backwardPropogation(self,learning_rate):
        self.dw = self.X.T.dot(self.dy)
        self.dx = self.dy.dot(self.weight.T)
        self.weight = self.weight - learning_rate * self.dw
        return self
    def getDx(self):
        return self.dx
    def getWeight(self):
        return self.weight
class model:
    def __init__(self):
        self.layers = []
        return 
    def addLayer(self,layer):
        self.layers.append(layer)
        if(len(self.layers) == 1):
            self.n = 1
            self.i = 1
        else:
            self.n += 1
        return self
    def fit0(self, X, Y, learning_rate):

        y = self.forwardPropogation(X)

        mse = MeanSquaredError()
        print(mse(Y, y).numpy())
        r = Y-y

        dy = 2*r/((r.shape[0])*(r.shape[1]))
        self.i = self.n-1
        while(self.i >= 1):
            dy = self.layers[self.i].setDy(dy).backwardPropogation(learning_rate).getDx()
            self.i = self.i-1

    def predict(self, X_prediction):
        return self.forwardPropogation(X_prediction)

        
    def forwardPropogation(self, X):
        y = self.layers[0].input(X).forwardPropogation().output()
        self.i = 1
        while(self.i < self.n):
            y = self.layers[self.i].input(y).forwardPropogation().output()
            self.i = self.i+1
        return y
