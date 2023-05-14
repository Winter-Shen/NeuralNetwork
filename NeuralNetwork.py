import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm
from keras.initializers import GlorotUniform


class MyLayer:
    def __init__(self, in_dim, out_dim, dropout = False, dropout_probability = None, dropout_lsh = False, function_num = None, table_num = 1):
        self.inDim = in_dim
        self.outDim = out_dim
        self.initializeWeight(GlorotUniform())
        self.rate = None
        self.dropout = dropout
        self.dropout_probability = dropout_probability
        self.dropout_lsh = dropout_lsh
        self.function_num = function_num
        self.function_num = table_num
        if(dropout):
            self.rate = 1-dropout_probability
        elif(dropout_lsh):
            self.rate = 1-(1-1/(2**function_num))**table_num
    def dropoutConfiguration(self):
        return int(self.dropout) + int(self.dropout_lsh)
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
    def forwardPropagation(self, x, prediction = False):
        self.x = x.astype(np.float64)
        # prediction 
        if(prediction):
            return self.x.dot(self.weight)
        # standard dropout
        if(self.dropout):
            self.mask = np.random.rand(self.outDim) > self.dropout_probability
        # LSH dropout
        elif(self.dropout_lsh):
            #self.rate = (1-1/(2**self.function_num))
            # Construct Hash table
            hashTable = HashTable(hash_size=self.function_num, dim=self.inDim)
            for i, r in enumerate(self.weight.T):
                hashTable[r] = i
            # Construct mask matrix
            self.mask = np.zeros((x.shape[0],self.outDim), np.int8)
            for i, r in enumerate(x):
                keep = hashTable[r]
                self.mask[i][keep] = 1
        # No dropout
        else:
            return self.x.dot(self.weight)
        return (self.x.dot(self.weight)*self.mask)/self.rate
    def backwardPropagation(self,dy):
        if self.rate is not None:
            dy = (dy*self.mask)/(self.rate)
        self.dw = self.x.T.dot(dy)
        dx = dy.dot(self.weight.T)
        self.weight = self.weight - self.learningRate * self.dw
        return dx
    def reset(self):
        self.initializeWeight(GlorotUniform())
    def getWeight(self):
        return self.weight


class MyModel:
    def __init__(self):
        self.layers = []
        return 
    def addLayer(self,layer):
        self.layers.append(layer)
        self.n = 1 if(len(self.layers) == 1) else self.n+1
        return self
    def __fit(self, x, y):
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
    def fit(self, X, Y, learning_rate, epoches, batch_size, progress = True):
        #Generate Batches
        batches = self.__getBatches(X, Y, batch_size)
        l = len(batches)

        for layer in self.layers:
            layer.setLearningRate(learning_rate)

        for i in range(epoches):
            batchesP = tqdm(range(l), bar_format='{desc}{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}', disable=not progress)
            cm = 0
            for j in batchesP:
                [loss, accuracy] = self.__fit(batches[j][0], batches[j][1])
                cm = (cm * j+accuracy)/(j+1)
                batchesP.set_description("Epoch %2d" % (i+1))
                batchesP.set_postfix_str("Accuracy: %.4f, Loss: %.4f;" % (round(cm, 4), round(loss, 4)))
    def predict(self, x):
        for l in self.layers:
            x = l.forwardPropagation(x, prediction = True)
        return x
    def reset(self):
        for l in self.layers:
            l.reset()
    def __getBatches(self, X, Y, batch_size):
        indices = np.arange(len(X))
        batches = [(X[indices[i:i+batch_size]], Y[indices[i:i+batch_size]]) for i in range(0, len(X), batch_size)]
        return batches

    
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

def dropout(value, LSH = False):
    return dropoutLSH(value) if(LSH) else dropout_normal(value)

class dropout_normal:
    def __init__(self, rate):
        self.rate = rate
    def forwardPropagation(self, x):
        k = np.random.rand(x.shape[1])
        self.mask = k > self.rate
        return (x*self.mask)/(1-self.rate)
    def backwardPropagation(self, dy):
        return (dy * self.mask)/(1-self.rate)

class dropoutLSH:
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
