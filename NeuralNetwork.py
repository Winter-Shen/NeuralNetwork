import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm
from keras.initializers import GlorotUniform


class MyLayer:
    def __init__(self, in_dim, out_dim):
        self.inDim = in_dim
        self.outDim = out_dim
        self.initializeWeight(GlorotUniform())
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.inDim, self.outDim)).numpy())
    def setWeight(self, weight):
        if(weight.shape[0] != self.inDim or weight.shape[1]!= self.outDim):
            return None
        else:
            self.weight = weight.astype(np.float64)
            return self
    def setLearningRate(self, learning_rate):
        self.learningRate = learning_rate
    def forwardPropagation(self, x):
        self.x = x.astype(np.float64)
        return self.x.dot(self.weight)
    def backwardPropagation(self,dy):
        self.dw = self.x.T.dot(dy)
        dx = dy.dot(self.weight.T)
        self.weight = self.weight - self.learningRate * self.dw
        return dx
    def getWeight(self):
        return self.weight

class dropout:
    def __init__(self, rate):
        self.rate = rate
    def forwardPropagation(self, x):
        k = np.random.rand(x.shape[1])
        self.mask = k > self.rate
        return (x*self.mask)/(1-self.rate)
    def backwardPropagation(self, dy):
        return (dy * self.mask)/(1-self.rate)

class MyModel:
    def __init__(self):
        self.layers = []
        self.dataLayer = []
        return 
    def addLayer(self,layer):
        self.layers.append(layer)
        self.n = 1 if(len(self.layers) == 1) else self.n+1
        if(isinstance(layer, MyLayer)):
            self.dataLayer.append(self.n-1)
        return self
    def fit(self, X, Y, learning_rate, epoches, batch_size):
        metrics = CategoricalAccuracy()
        #Generate Batches
        batches = self.getBatches(X, Y, batch_size)
        l = len(batches)

        for i in self.dataLayer:
            self.layers[i].setLearningRate(learning_rate)

        for i in range(epoches):
            batchesP = tqdm(range(l), bar_format="{l_bar}")
            cm = 0
            for j in batchesP:
                [loss, accuracy] = self.fit0(batches[j][0], batches[j][1])
                cm = (cm * j+accuracy)/(j+1)
                batchesP.set_description("Epoch %2d: %3d/%3d; Accuracy: %.4f; Loss: %.4f" % (i+1, j+1, l, cm, loss))
    def fit0(self, x, y):
        # Forwardpropogation
        for l in self.layers:
            x = l.forwardPropagation(x)
        y_hat = x
        mse = MeanSquaredError()
        metrics = CategoricalAccuracy()
        loss = mse(y, y_hat).numpy()
        metrics.update_state(y, y_hat)
        accuracy = metrics.result().numpy()

        r = y_hat-y
        dy = 2*r/((r.shape[0])*(r.shape[1]))
        # backwardpropogation
        for l in range(self.n-1, -1, -1):
            dy = self.layers[l].backwardPropagation(dy)
        return([loss, accuracy])
    def getBatches(self, X, Y, batch_size):
        indices = np.arange(len(X))
        batches = [(X[indices[i:i+batch_size]], Y[indices[i:i+batch_size]]) for i in range(0, len(X), batch_size)]
        return batches
    def predict(self, x):
        for i in self.dataLayer:
            x = self.layers[i].forwardPropagation(x)
        return x

'''
class MyLayer1:
    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim
        self.prob = 0
        self.initializeWeight(GlorotUniform())
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.inDim, self.outDim)).numpy())
    def setWeight(self, weight):
        if(weight.shape[0] != self.inDim or weight.shape[1]!= self.outDim):
            return None
        else:
            self.weight = weight.astype(np.float64)
            return self
    def dropout(self, prob):
        self.prob = prob
        return self
    def setMask(self, random_sequence):
        self.mask = random_sequence > self.prob
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
        return (self.Y*self.mask)/(1-self.prob)
    def setDy(self, dy):
        if(dy.shape[0] != self.size or dy.shape[1]!= self.outDim):
            return None
        else:
            self.dy = (dy * self.mask)/(1-self.prob)
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

    def fit0(self, X, Y, learning_rate, random_sequences):
        # Set masks for each layer
        for i in range(self.n):
            self.layers[i].setMask(random_sequences[i])
        # Forwardpropogation
        y = self.forwardPropogation(X)

        mse = MeanSquaredError()
        metrics = CategoricalAccuracy()

        loss = mse(Y, y).numpy()
        metrics.update_state(Y, y)
        accuracy = metrics.result().numpy()

        r = y-Y
        dy = 2*r/((r.shape[0])*(r.shape[1]))
        
        i = self.n-1
        while(i >= 0):
            dy = self.layers[i].setDy(dy).backwardPropogation(learning_rate).getDx()
            i = i-1

        return([loss, accuracy])
    def fit(self, X, Y, learning_rate, epoches, batch_size):
        metrics = CategoricalAccuracy()
        mse = MeanSquaredError()
        #Generate Batches
        batches = self.getBatches(X, Y, batch_size)
        l = len(batches)

        for i in range(epoches):
            batchesP = tqdm(range(l), bar_format="{l_bar}")
            for j in batchesP:

                matrix = [np.random.rand(self.layers[k].outDim) for k in range(self.n)]    
                [loss, accuracy] = self.fit0(batches[j][0], batches[j][1], learning_rate, matrix)
                batchesP.set_description("Epoch %2d: %3d/%3d; Accuracy: %.3f; Loss: %.3f" % (i+1, j+1, l, accuracy, loss))

    def predict(self, X_prediction):
        # Set elements in mask to True
        for k in range(self.n):
            self.layers[k].setMask(np.ones(self.layers[k].outDim))
        return self.forwardPropogation(X_prediction)
    def forwardPropogation(self, X):
        y = X
        i = 0
        while(i < self.n):
            y = self.layers[i].input(y).forwardPropogation().output()
            i = i+1
        return y
    def getBatches(self, X, Y, batch_size):
        indices = np.arange(len(X))
        #np.random.shuffle(indices)
        batches = [(X[indices[i:i+batch_size]], Y[indices[i:i+batch_size]]) for i in range(0, len(X), batch_size)]
        return batches
'''