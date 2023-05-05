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

class dropoutS:
    def __init__(self, size):
        self.size = size
        self.rate = 1/(2**size)
    def constructHashTable(self,dim,weight,x):
        # Construct Hash table
        self.hashTable = HashTable(hash_size=self.size, dim=dim)
        for i, r in enumerate(weight.T):
            self.hashTable[r] = i
        # Construct mask matrix
        self.mask = np.zeros((x.shape[0],weight.shape[1]), np.int8)
        for i, r in enumerate(x):
            keep = self.hashTable[r]
            self.mask[i][keep] = 1
    def forwardPropagation(self, x):
        return (x*self.mask)/(self.rate)
    def backwardPropagation(self, dy):
        return (dy*self.mask)/(self.rate)

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
    def fitS(self, X, Y, learning_rate, epoches, batch_size):
        #Generate Batches
        batches = self.getBatches(X, Y, batch_size)
        l = len(batches)

        for i in self.dataLayer:
            self.layers[i].setLearningRate(learning_rate)

        for i in range(epoches):
            batchesP = tqdm(range(l), bar_format="{l_bar}")
            #batchesP = range(l)
            cm = 0
            for j in batchesP:
                [loss, accuracy] = self.fitS0(batches[j][0], batches[j][1])
                cm = (cm * j+accuracy)/(j+1)
                batchesP.set_description("Epoch %2d: %3d/%3d; Accuracy: %.4f; Loss: %.4f" % (i+1, j+1, l, cm, loss))
    def fitS0(self, x, y):
        # Forwardpropogation
        xx = x 
        for l in self.layers:
            if(isinstance(l, dropoutS)):
                l.constructHashTable(dim, w, xx)
                xx = x
                x=l.forwardPropagation(x)
            else:
                x=l.forwardPropagation(x)
                dim = l.inDim
                w = l.weight
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
    
class HashTable:
    def __init__(self, hash_size, dim):
        self.hash_size = hash_size
        self.inp_dimensions = dim
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, dim)
        
    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])

