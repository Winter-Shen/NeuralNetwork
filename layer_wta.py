from layer import Layer
import numpy as np

class LayerWTA(Layer):
    def __init__(self, in_dim: int, out_dim: int, top_k: float = None):
        super().__init__(in_dim, out_dim)
        self.k = int(self._out_dim * (1 - top_k))


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