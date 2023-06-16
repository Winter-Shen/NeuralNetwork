from layer import Layer
import numpy as np

class LayerLSH(Layer):
    def __init__(self, in_dim: int, out_dim: int, function_num:int = 1, table_num:int = 1):
        super().__init__(in_dim, out_dim)

        self._function_num = function_num
        self._table_num = table_num
        self._hash_range = range(2**function_num)
        self.__constructHashTable()


    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        active_set = set()
        for i in range(self._table_num):
            hash_vector = np.dot(x, self._projections[i].T) > 0
            hash_value = self.__binary_vector_to_integer(hash_vector)[0]
            active_set = self._hash_tables[i][hash_value] | active_set
        active_set_idx = list(active_set)

        self._mask = np.zeros(self._out_dim, dtype=bool)
        self._mask[active_set_idx] = True
        return np.dot(x, self._weight)*self._mask

    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self, dy: np.ndarray) -> np.ndarray:
        dy = (dy*self._mask)

        dx = np.dot(dy, self._weight.T)
        dw = np.dot(self._x.T, dy)
        self._weight = self._weight - self._learning_rate * dw

        for i in range(self._table_num):
            hash_vectors = np.dot(self._projections[i], self._weight) > 0
            hash_values = np.apply_along_axis(self.__binary_vector_to_integer, axis=0, arr=hash_vectors)

            for idx, value in enumerate(hash_values):
                old_hash_value = self._hash_values[i][idx]
                self._hash_tables[i][old_hash_value].remove(idx)
                self._hash_tables[i][value].add(idx)
            self._hash_values[i] = hash_values

        return dx

    def predict(self, X:np.ndarray) -> np.ndarray:
        y = np.dot(X, self._weight)

        mask = np.zeros(y.shape, dtype=bool)
        for i in range(self._table_num):
            hash_vectors = np.dot(X, self._projections[i].T) > 0
            mask = np.apply_along_axis(self.__binary_vector_by_hash_vector, axis=1, arr=hash_vectors, args=i) | mask

        return y*mask

    def __binary_vector_to_integer(self, vector):
        value = 0
        for b, s in enumerate(vector):
            value = value^(s<<b)
        return value
    
    def __binary_vector_by_hash_vector(self, vector, args):
        mask = np.zeros(self._out_dim, dtype=bool)
        hash_value = self.__binary_vector_to_integer(vector)
        active_set = list(self._hash_tables[args][hash_value])
        mask[active_set] =  True
        return mask

    def __constructHashTable(self):
        self._projections = []
        self._hash_tables = []
        self._hash_values = np.empty((self._table_num, self._out_dim), dtype=np.int8)
        for i in range(self._table_num):
            projection = np.random.randn(self._function_num, self._in_dim)
            hash_vectors = np.dot(projection, self._weight) > 0
            hash_values = np.apply_along_axis(self.__binary_vector_to_integer, axis=0, arr=hash_vectors)
            hash_table = [set(np.where(hash_values == values)[0]) for values in self._hash_range]

            self._hash_values[i] = hash_values
            self._hash_tables.append(hash_table)
            self._projections.append(projection)