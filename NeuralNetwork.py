import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm

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
        self.prob = 1.1
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.inDim, self.outDim)).numpy())
    def setWeight(self, weight):
        if(weight.shape[0] != self.inDim or weight.shape[1]!= self.outDim):
            return None
        else:
            self.weight = weight.astype(np.float64)
            return self
    def activeNode(self, prob):
        self.prob = prob
        return self
    def input(self, data):
        if(data.shape[1] != self.inDim):
            return None
        else:
            self.X = data.astype(np.float64)
            self.size = data.shape[0]

            r = np.random.rand(self.size, self.outDim)
            r = r < self.prob
            self.r = r

            return self
    def forwardPropogation(self):
        self.Y = self.X.dot(self.weight)
        return self
    def output(self):
        return (self.Y*self.r)/self.prob
    def setDy(self, dy):
        if(dy.shape[0] != self.size or dy.shape[1]!= self.outDim):
            return None
        else:
            self.dy = (dy * self.r)/self.prob
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
class MyModel:
    def __init__(self):
        self.layers = []
        return 
    def addLayer(self,layer):
        self.layers.append(layer)
        if(len(self.layers) == 1):
            self.n = 1
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
        i = self.n-1

        while(i >= 1):
            dy = self.layers[i].setDy(dy).backwardPropogation(learning_rate).getDx()
            i = i-1
        return([mse(Y, y).numpy(), metrics.result().numpy()])
    def fit(self, X, Y, learning_rate, epoches, batch_size):

        batches = self.getBatches(X, batch_size)
        '''
        l = len(batches)
        for i in range(epoches):
            for batch in tqdm(range(l), dynamic_ncols=True):
                [loss, accuracy] = self.fit0(batches[batch], Y, learning_rate)
                print("Epoch %2d: loss: %.3f - accuracy: %.3f" % (i+1, loss, accuracy))
        '''
        for i in range(epoches):
            for batch in batches:
                [loss, accuracy] = self.fit0(batch, Y, learning_rate)
                print("Epoch %2d: loss: %.3f - accuracy: %.3f" % (i+1, loss, accuracy))

    def predict(self, X_prediction):
        return self.forwardPropogation(X_prediction)
    def forwardPropogation(self, X):
        y = self.layers[0].input(X).forwardPropogation().output()
        i = 1
        while(i < self.n):
            y = self.layers[i].input(y).forwardPropogation().output()
            i = i+1
        return y
    def getBatches(self, X, batch_size):
        batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        return batches