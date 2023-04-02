import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm
from keras.initializers import GlorotUniform

class InputLayer:
    def __init__(self, dim):
        self.dim = dim
        self.livelyNode = list(range(dim))
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
        self.r = np.ones(outDim)
        self.initializeWeight(GlorotUniform())
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
        r = np.random.rand(self.outDim)
        #r = np.random.rand(self.size, self.outDim)
        r = r < prob
        self.r = r
        return self
    def dropout(self, prob):
        self.prob = prob
        return self
    def setR(self, r):
        self.r = r < self.prob
        return self
    def input(self, data):
        if(data.shape[1] != self.inDim):
            return None
        else:
            self.X = data.astype(np.float64)
            self.size = data.shape[0]

            return self
    def forwardPropogation(self):
        self.Y = self.X.dot(self.weight)
        return self
    def output(self):
        return (self.Y*self.r)#/self.prob
    def setDy(self, dy):
        if(dy.shape[0] != self.size or dy.shape[1]!= self.outDim):
            return None
        else:
            self.dy = (dy * self.r)#/self.prob
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
    def fit0(self, X, Y, learning_rate, r):


        y = self.forwardPropogation(X, r)

        mse = MeanSquaredError()
        metrics = CategoricalAccuracy()
        r = y-Y

        dy = 2*r/((r.shape[0])*(r.shape[1]))
        i = self.n-1

        while(i >= 1):
            dy = self.layers[i].setDy(dy).backwardPropogation(learning_rate).getDx()
            i = i-1

        y = self.predict(X)
        metrics.update_state(Y, y)
        return([mse(Y, y).numpy(), metrics.result().numpy()])
    def fit(self, X, Y, learning_rate, epoches, batch_size):
        batches = self.getBatches(X, Y, batch_size)
        batch_number = X.shape[0]//batch_size + 1
        self.batch_number = batch_number

        r = [np.random.rand(batch_number, self.layers[k].outDim) for k in range(1, self.n)]

        l = len(batches)
        for i in range(epoches):
            batchesP = tqdm(range(l), bar_format="{l_bar}")
            for j in batchesP:
                
                [loss, accuracy] = self.fit0(batches[j][0], batches[j][1], learning_rate, [r[k][j]  for k in range(0, self.n-1)])
                batchesP.set_description("Epoch %2d: %3d/%3d; Accuracy %.3f" % (i+1, j+1, batch_number, accuracy))  
    def predict(self, X_prediction):
        r = [np.zeros((self.layers[k].outDim)) for k in range(1, self.n)]
        return self.forwardPropogation(X_prediction, r)
    def forwardPropogation(self, X, r):
        #print(len(r))
        y = self.layers[0].input(X).forwardPropogation().output()
        i = 1
        while(i < self.n):
            y = self.layers[i].setR(r[i-1]).input(y).forwardPropogation().output()
            i = i+1
        return y
    def getBatches(self, X, Y, batch_size):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        batches = [(X[indices[i:i+batch_size]], Y[indices[i:i+batch_size]]) for i in range(0, len(X), batch_size)]
        return batches