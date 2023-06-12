from layer import Layer
import numpy as np

class LayerDropout(Layer):
    def __init__(self, in_dim: int, out_dim: int, dropout_probability: float = None):
        super().__init__(in_dim, out_dim)
        self._dropout_probability = dropout_probability


    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._mask = np.random.rand(self._out_dim) > self._dropout_probability
        return np.dot(x, self._weight)*self._mask


    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self, dy: np.ndarray) -> np.ndarray:
        dy = (dy*self._mask)

        dx = np.dot(dy, self._weight.T)
        dw = np.dot(self._x.T, dy)
        self._weight = self._weight - self._learning_rate * dw
        return dx