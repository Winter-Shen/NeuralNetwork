import numpy as np
from keras.initializers import GlorotUniform

class Layer:
    def __init__(self, in_dim: int, out_dim: int):
        self._in_dim = in_dim # number of dimesnions in the input set
        self._out_dim = out_dim # number of dimesnions in the output set -- number of neurals
        self.initializeWeight(GlorotUniform())
            

    def trainingSettings(self, learning_rate: float) -> None:
        self._learning_rate = learning_rate

    # Initialize weight matrix by a initializer of keras.initializers
    def initializeWeight(self, initializer) -> None:
        self.setWeight(initializer((self._in_dim, self._out_dim)).numpy())

    # Initialize weight matrix by a numpy array
    def setWeight(self, weight: np.ndarray)  -> None:
        self._weight = weight.astype(np.float64)

    # Take input set x and return output set after forward propagation
    def forwardPropagation(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.dot(x, self._weight)

    def predict(self, X) -> np:
        return np.dot(X, self._weight)

    # Take derivatives of output set and return derivatives of inputset set after backward propagation
    def backwardPropagation(self, dy: np.ndarray) -> np.ndarray:
        dx = np.dot(dy, self._weight.T)
        dw = np.dot(self._x.T, dy)
        self._weight = self._weight - self._learning_rate * dw
        return dx