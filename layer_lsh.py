from layer import Layer
import numpy as np

class LayerLSH(Layer):
    def __init__(self, in_dim: int, out_dim: int, function_num:int = 1, table_num:int = 1):
        super().__init__(in_dim, out_dim)

        self.function_num = function_num
        self.table_num = table_num
        self.buckect_num = 2**function_num
        self.rate = 1-(1-1/(2**function_num))**table_num

        #self.hash_tables = [[set()]*self.buckect_num]*table_num # Hash tables 
        self.hash_tables = [[set() for j in range(self.buckect_num)] for i in range(table_num)]
        self.hash_values = [[0 for j in range(out_dim)] for i in range(table_num)] # Lists to hold hash values to avoid repeating computing hash values
        self.projections = [np.random.randn(in_dim + 1, function_num) for i in range(table_num)] # Projection vector, the component of hash function
        self.__constructHashTable()

        self._projection = np.random.randn(function_num, in_dim)
        hash_vectors = np.dot(self._projection, self._weight) > 0
        hash_values = np.apply_along_axis(self.__binary_vector_to_integer, axis=0, arr=hash_vectors)
        unique_values = np.unique(hash_values)
        self.hash_table = {values: set(np.where(hash_values == values)[0]) for values in unique_values}


    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x: np.ndarray) -> np.ndarray:
        self._x = x

        score = np.dot(x, self._weight) # inner product which can be treated as score or ouput
        indics = np.argsort(score)[0] # indexes of ascending values
        self.tail_index = indics[0:self.k] # indexes of elements with low scores 
        score[:, self.tail_index] = 0
        return score

    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self, dy: np.ndarray) -> np.ndarray:
        dy[:, self.tail_index] = 0

        dx = np.dot(dy, self._weight.T)
        dw = np.dot(self._x.T, dy)
        self._weight = self._weight - self._learning_rate * dw
        return dx

    def predict(self, X:np.ndarray) -> np.ndarray:
        score = np.dot(X, self._weight)

        indics = np.argsort(score)
        tail_index = indics[:,0:self.k]

        score[[[i] for i in range(X.shape[0])], tail_index] = 0
        return score

    def __binary_vector_to_integer(self, vector):
        value = 0
        for b, s in enumerate(vector):
            value = value^(s<<b)
        return value