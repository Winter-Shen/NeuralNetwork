import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy

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
class MyLayer:
    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim
        self.livelyCol = list(range(outDim))
        self.livelyRow = list(range(inDim))
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.inDim, self.outDim)).numpy())
    def setWeight(self, weight):
        if(weight.shape[0] != self.inDim or weight.shape[1]!= self.outDim):
            return None
        else:
            self.weight = weight.astype(np.float64)
            return self
    def activeNode(self, livelyRow, livelyCol, prob):
        '''
        self.livelyCol = livelyCol
        self.livelyRow = livelyRow
        self.inDim = len(livelyRow)
        self.outDim = len(livelyCol)
        '''
        r = np.random.rand(self.inDim, self.outDim)
        r = r < prob
        self.prob = prob
        self.r = r
    def input(self, data):
        if(data.shape[1] != self.inDim):
            return None
        else:
            self.X = data.astype(np.float64)
            self.size = data.shape[0]
            return self
    def forwardPropogation(self):
        #self.Y = self.X.dot(self.weight[np.transpose([self.livelyRow]),self.livelyCol])
        self.Y = self.X.dot(self.weight)
        return self
    def output(self):
        return (self.Y*self.r)/prob
    def setDy(self, dy):
        if(dy.shape[0] != self.size or dy.shape[1]!= self.outDim):
            return None
        else:
            self.dy = dy
            return self
    def backwardPropogation(self,learning_rate):
        #livelyRow = np.transpose([self.livelyRow])
        self.dw = self.X.T.dot(self.dy)
        #self.dx = self.dy.dot(self.weight[livelyRow,self.livelyCol].T)
        self.dx = self.dy.dot(self.weight[livelyRow,self.livelyCol].T)
        self.weight[livelyRow,self.livelyCol] = self.weight[livelyRow,self.livelyCol] - learning_rate * self.dw
        return self
    def getDx(self):
        return self.dx
    def getWeight(self):
        return self.weight
class MyModel:
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
        metrics = CategoricalAccuracy()
        metrics.update_state(Y, y)
        r = y-Y

        dy = 2*r/((r.shape[0])*(r.shape[1]))
        self.i = self.n-1
        while(self.i >= 1):
            dy = self.layers[self.i].setDy(dy).backwardPropogation(learning_rate).getDx()
            self.i = self.i-1
        return([mse(Y, y).numpy(), metrics.result().numpy()])
    def fit(self, X, Y, learning_rate, epochs):
        for i in range(epochs):
            [loss, accuracy] = self.fit0(X, Y, learning_rate)
            print("Epoch %2d: loss: %.3f - accuracy: %.3f" % (i+1, loss, accuracy))
    def predict(self, X_prediction):
        return self.forwardPropogation(X_prediction)
    def forwardPropogation(self, X):
        y = self.layers[0].input(X).forwardPropogation().output()
        self.i = 1
        while(self.i < self.n):
            y = self.layers[self.i].input(y).forwardPropogation().output()
            self.i = self.i+1
        return y
    def selectNodes(self, active_node):
        rowIndex = list(range(self.layers[0].dim))
        for i, l in enumerate(active_node):
            colIndex = l
            self.layers[i+1].activeNode(rowIndex, colIndex)
            rowIndex = l
        ll = self.layers[self.n-1]
        ll.activeNode(rowIndex, ll.livelyCol)