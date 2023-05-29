import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm
from keras.initializers import GlorotUniform

class MyLayer:
    def __init__(self, in_dim, out_dim, dropout = False, dropout_probability = None, dropout_lsh = False, function_num = None, table_num = 1):
        self.in_dim = in_dim # number of dimesnions in the input set
        self.out_dim = out_dim # number of dimesnions in the output set -- number of neurals
        self.initializeWeight(GlorotUniform()) 
        self.rate = None # drop out rate
        self.dropout = dropout # flag to drop out
        self.dropout_lsh = dropout_lsh # flag to drop out with LSH method
        if(dropout and dropout_probability is not None):
            self.dropout_probability = dropout_probability
            self.rate = 1-dropout_probability
        elif(dropout_lsh and function_num is not None):
            self.function_num = function_num
            self.table_num = table_num
            self.buckect_num = 2**function_num
            self.rate = 1-(1-1/(2**function_num))**table_num

            self.hash_tables = [[set()]*self.buckect_num]*table_num # Hash tables 
            self.hash_values = [[0]*out_dim]*table_num # Lists to hold hash values to avoid repeating computing hash values
            self.projections = [np.random.randn(in_dim, function_num)]*table_num # Projection vector, the component of hash function

            self.__constructHashTable()
    def dropoutConfiguration(self):
        code = 0
        code = code^(self.dropout<<1)
        code = code^(self.dropout_lsh)
        return code
    # Initialize weight matrix by a initializer of keras.initializers
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.in_dim, self.out_dim)).numpy())
    # Initialize weight matrix by a numpy array
    def setWeight(self, weight):
        if(weight.shape[0] != self.in_dim or weight.shape[1]!= self.out_dim):
            return None
        else:
            self.weight = weight.astype(np.float64)
            return self
    # Set learning rate
    def setLearningRate(self, learning_rate):
        self.learningRate = learning_rate 
    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x, prediction = False):
        self.x = x.astype(np.float64)
        # prediction 
        if(prediction):

            if(self.dropout_lsh):
                self.__collectActiveSet()
                return (self.x.dot(self.weight)*self.mask)/self.rate

            return self.x.dot(self.weight)
        # standard dropout
        if(self.dropout and self.dropout_probability is not None):
            self.mask = np.random.rand(self.out_dim) > self.dropout_probability
        # LSH dropout
        elif(self.dropout_lsh):
            self.__collectActiveSet()
        # No dropout
        else:
            return self.x.dot(self.weight)
        return (self.x.dot(self.weight)*self.mask)/self.rate
    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self,dy):
        if self.rate is not None:
            dy = (dy*self.mask)/(self.rate)
        dx = dy.dot(self.weight.T)
        self.dw = self.x.T.dot(dy)
        self.weight = self.weight - self.learningRate * self.dw
        if(self.dropout_lsh):
            self.__updateHashTables()
        return dx
    def __constructHashTable(self):
        for idx, w in enumerate(self.weight.T):

            for i in range(self.table_num):
                hash_value = self.__computeFingerprint(w, self.projections[i])
                self.hash_values[i][idx] = hash_value
                self.hash_tables[i][hash_value].add(idx)
    def __collectActiveSet(self):
        self.mask = np.zeros((self.x.shape[0],self.out_dim), np.int8)
        for i, x in enumerate(self.x):
            acitive_set = set()
            for t,projection in enumerate(self.projections):
                fp = self.__computeFingerprint(x, projection)
                acitive_set = self.hash_tables[t][fp] | acitive_set
            self.mask[i][list(acitive_set)] = 1
    def __updateHashTables(self):
        for idx, w in enumerate(self.weight.T):
            for i in range(self.table_num):
                self.hash_tables[i][self.hash_values[i][idx]].remove(idx)

                hash_value = self.__computeFingerprint(w, self.projections[i])        
                self.hash_values[i][idx] = hash_value
                self.hash_tables[i][hash_value].add(idx)

    def __computeFingerprint(self, vector, projection):
        hash_vector = (np.dot(vector, projection) > 0).astype('int8')
        hash_value = 0
        for b, s in enumerate(hash_vector):
            hash_value = hash_value^(s<<b)
        return hash_value
    def __computeFingerprints(self, vector):
        return [self.__computeFingerprint(vector, projection) for projection in self.projections]
    def reset(self):
        self.initializeWeight(GlorotUniform())
        if(self.dropout_lsh):
            self.hash_tables = [[set()]*self.buckect_num]*self.table_num
            self.hash_values = [[0]*self.out_dim]*self.table_num
            self.projections = [np.random.randn(self.in_dim, self.function_num)]*self.table_num
            self.__constructHashTable()

class MyModel:
    def __init__(self):
        self.layers = []
        return 
    def addLayer(self,layer):
        self.layers.append(layer)
        self.n = 1 if(len(self.layers) == 1) else self.n+1
        return self
    # A forward and backward cycle
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