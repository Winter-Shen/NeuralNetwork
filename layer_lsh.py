from layer import Layer
from typing import Union
import numpy as np

class LayerLSH(Layer):
    def __init__(self, in_dim: int, out_dim: int, function_num:int = 1, table_num:int = 1, previous:bool = False, next:bool = False):
        super().__init__(in_dim, out_dim)

        self._previous = previous
        self._next = next
        self._function_num = function_num
        self._table_num = table_num
        self._hash_range = range(2**function_num)
        self.__constructHashTable()


    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x: Union[list, np.ndarray]) -> Union[list, np.ndarray]:

        self._active_sets = [self._hash_tables[i][np.packbits((np.dot(x[0] if self._previous else x, self._projections[i].T) > 0)[0], bitorder='little')[0]] for i in range(self._table_num)]
        
        if(self._previous):
            x[0][0][x[1]] = 0
            self._x = x[0]
        else:
            self._x = x




        self._mask = np.ones(self._out_dim, dtype=bool)
        self._mask[list(set().union(*self._active_sets))] = False

        y = np.dot(self._x, self._weight)
        #y[0][self._mask] = 0

        if(self._next):
            return [y, self._mask]
        else:
            y[0][self._mask] = 0
            return y

    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self, dy: np.ndarray) -> np.ndarray:
        dy[0][self._mask] = 0

        dx = np.dot(dy, self._weight.T)
        dw = np.dot(self._x.T, dy)
        self._weight = self._weight - self._learning_rate * dw

        for i in range(self._table_num):
            active_set_idx  = list(self._active_sets[i])

            if(len(active_set_idx) == 0):
                continue

            hash_vectors = np.dot(self._projections[i], self._weight[:,active_set_idx]) > 0
            hash_values = np.packbits(hash_vectors, axis=0, bitorder='little')[0]
            old_hash_values = self._hash_values[i][active_set_idx]
            
            for j in range(len(active_set_idx)):
                self._hash_tables[i][old_hash_values[j]].remove(active_set_idx[j])
                self._hash_tables[i][hash_values[j]].add(active_set_idx[j])
            
            self._hash_values[i][active_set_idx] = hash_values
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
            #hash_values = np.apply_along_axis(self.__binary_vector_to_integer, axis=0, arr=hash_vectors)
            hash_values = np.packbits(hash_vectors, axis=0, bitorder='little')[0]
            
            hash_table = [set(np.where(hash_values == values)[0]) for values in self._hash_range]
            

            self._hash_values[i] = hash_values
            self._hash_tables.append(hash_table)
            self._projections.append(projection)