import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm
from keras.initializers import GlorotUniform

class MyLayer:
    def __init__(self, in_dim, out_dim, dropout = False, dropout_probability = None, dropout_lsh = False, function_num = None, table_num = 1):
        self.in_dim = in_dim # number of dimesnions in the input set
        self.out_dim = out_dim # number of dimesnions in the output set -- number of neurals
        self.rate = None # drop out rate
        self.dropout = dropout # flag to drop out
        self.dropout_lsh = dropout_lsh # flag to drop out with LSH method
        self.initializeWeight(GlorotUniform())
        if(dropout and dropout_probability is not None):
            self.dropout_probability = dropout_probability
            self.rate = 1-dropout_probability
        elif(dropout_lsh and function_num is not None):
            self.function_num = function_num
            self.table_num = table_num
            self.buckect_num = 2**function_num
            self.rate = 1-(1-1/(2**function_num))**table_num

            #self.hash_tables = [[set()]*self.buckect_num]*table_num # Hash tables 
            self.hash_tables = [[set() for j in range(self.buckect_num)] for i in range(table_num)]
            self.hash_values = [[0 for j in range(out_dim)] for i in range(table_num)] # Lists to hold hash values to avoid repeating computing hash values
            self.projections = [np.random.randn(in_dim + 1, function_num) for i in range(table_num)] # Projection vector, the component of hash function

            self.__constructHashTable()

    def globalSettings(self, n, batch_size, learning_rate):
        self.mask_X = np.full((n,self.out_dim), 0, dtype=bool)
        self.batch_size = np.append(np.full(n//batch_size, batch_size), n%batch_size)
        self.computed_mask_batch_flag = np.zeros(n//batch_size + 1, dtype=bool)
        self.learning_rate = learning_rate 


    def dropoutConfiguration(self):
        return 0^(self.dropout<<1)^(self.dropout_lsh)

    # Initialize weight matrix by a initializer of keras.initializers
    def initializeWeight(self, initializer):
        return self.setWeight(initializer((self.in_dim, self.out_dim)).numpy())

    # Initialize weight matrix by a numpy array
    def setWeight(self, weight):
        self.weight = weight.astype(np.float64)
        return self

    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x, batch_label = -1, prediction = False):
        self.batch = x.astype(np.float64)
        # prediction 
        if(prediction):

            if(self.dropout_lsh):
                self.__collectActiveSet(batch_label)
                return (self.batch.dot(self.weight)*self.mask)/self.rate
                #return (self.batch.dot(self.weight))

            return self.batch.dot(self.weight)
        # standard dropout
        if(self.dropout and self.dropout_probability is not None):
            self.mask = np.random.rand(self.out_dim) > self.dropout_probability
        # LSH dropout
        elif(self.dropout_lsh):
            self.__collectActiveSet(batch_label)
        # No dropout
        else:
            return self.batch.dot(self.weight)
        return (self.batch.dot(self.weight)*self.mask)/self.rate
        #return (self.batch.dot(self.weight)*self.mask)

    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self,dy):
        if self.rate is not None:
            dy = (dy*self.mask)/(self.rate)
            #dy = (dy*self.mask)
        dx = dy.dot(self.weight.T)
        self.dw = self.batch.T.dot(dy)
        self.weight = self.weight - self.learning_rate * self.dw
        if(self.dropout_lsh):
            self.__updateHashTables()
        return dx

    def __collectActiveSet(self, batch_label):
        if(batch_label != -1): # Input for this training cycle is a batch

            idx_range = list(range(batch_label*(self.batch_size[batch_label]),batch_label*(self.batch_size[batch_label])+self.batch_size[batch_label]))

            if(self.computed_mask_batch_flag[batch_label]): # The mask vectors have alraedy been computed.
                self.mask = self.mask_X[idx_range]
            else: # The mask vectors have not been computed.
                self.mask = np.zeros((self.batch_size[batch_label],self.out_dim), dtype=bool)
                for i, x in enumerate(self.batch):
                    
                    # Fix x
                    norm = np.sqrt(np.dot(x, x.T))
                    new_x = np.append(x/norm, 0)

                    acitive_set = set()
                    for t,projection in enumerate(self.projections):
                        fp = self.__computeFingerprint(new_x, projection)
                        acitive_set = self.hash_tables[t][fp] | acitive_set
                    self.mask[i][list(acitive_set)] = True

                self.mask_X[idx_range] = self.mask
                self.computed_mask_batch_flag[batch_label] = True
        else: # Input for this training cycle is the complete X. Often run when it is prediction and all the mask vectors should be computed 
            self.mask = np.zeros((self.batch.shape[0],self.out_dim), dtype=bool)
            for i, x in enumerate(self.batch):

                # Fix x
                norm = np.sqrt(np.dot(x, x.T))
                new_x = np.append(x/norm, 0)

                acitive_set = set()
                for t,projection in enumerate(self.projections):
                    fp = self.__computeFingerprint(new_x, projection)
                    acitive_set = self.hash_tables[t][fp] | acitive_set
                self.mask[i][list(acitive_set)] = True

    def __constructHashTable(self):
        self.__maxNormWeight()
        for idx, w in enumerate(self.weight.T):

            for i in range(self.table_num):

                # Fix weight vector
                weight = w/self.max_norm
                norm = self.norm[idx]
                e = np.sqrt(np.abs(1-(norm/self.max_norm)**2))
                new_w = np.append(weight, e)
                
                hash_value = self.__computeFingerprint(new_w, self.projections[i])
                self.hash_values[i][idx] = hash_value
                self.hash_tables[i][hash_value].add(idx)

    def __updateHashTables(self):
        self.__maxNormWeight()
        for idx, w in enumerate(self.weight.T):
            for i in range(self.table_num):

                # Fix weight vector
                weight = w/self.max_norm
                norm = self.norm[idx]
                e = np.sqrt(1-(norm/self.max_norm)**2)
                new_w = np.append(weight, e)

                # Remove the weight connection in the hash table
                self.hash_tables[i][self.hash_values[i][idx]].remove(idx)

                # Compute hash value and update the weight connection
                hash_value = self.__computeFingerprint(new_w, self.projections[i])        
                self.hash_values[i][idx] = hash_value
                self.hash_tables[i][hash_value].add(idx)

    def __maxNormWeight(self):
        self.norm = np.array([np.sqrt(np.dot(w, w.T)) for w in self.weight.T], dtype=np.float64)
        self.max_norm = np.max(self.norm)   


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

            self.hash_tables = [[set() for j in range(self.buckect_num)] for i in range(self.table_num)]
            self.hash_values = [[0 for j in range(self.out_dim)] for i in range(self.table_num)] # Lists to hold hash values to avoid repeating computing hash values
            self.projections = [np.random.randn(self.in_dim + 1, self.function_num) for i in range(self.table_num)] # Projection vector, the component of hash function
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
    def __fit(self, x, y, batch_label):
        # Forwardpropogation
        for l in self.layers:
            x = l.forwardPropagation(x, batch_label)

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
            #layer.setLearningRate(learning_rate)
            #layer.configureBatchParameter(X.shape[0], batch_size)
            layer.globalSettings(X.shape[0], batch_size, learning_rate)

        for i in range(epoches):
            batchesP = tqdm(range(l), bar_format='{desc}{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}', disable=not progress)
            cm = 0
            for j in batchesP:
                [loss, accuracy] = self.__fit(batches[j][0], batches[j][1], j)
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