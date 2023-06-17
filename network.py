from layer import Layer

import numpy as np
from keras.losses import MeanSquaredError
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm
import time

class Network:
    def __init__(self):
        self.__layers = []

    def addLayer(self, layer: Layer) -> None:
        self.__layers.append(layer)
        self.__n = 1 if(len(self.__layers) == 1) else self.__n+1

    # A forward and backward cycle
    def __fit(self, x, y) -> float:
        # Forwardpropogation
        for l in self.__layers:
            x = l.forwardPropagation(x)

        y_hat = x
        metrics = CategoricalAccuracy()
        metrics.update_state(y, y_hat)
        accuracy = metrics.result().numpy()

        r = y_hat-y
        dy = 2*r/(r.shape[1])
        # backwardpropogation
        for l in range(self.__n-1, -1, -1):
            dy = self.__layers[l].backwardPropagation(dy)
        return(accuracy)

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float, epochs: int, progress: bool = True) -> list:
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)

        for layer in self.__layers:
            layer.trainingSettings(learning_rate)

        epoch_accuracy = [0 for i in range(epochs)]
        epoch_time = [0 for i in range(epochs)]
        for i in range(epochs):
            
            singleP = tqdm(range(X.shape[0]), bar_format='{desc}{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}', disable=not progress)
            cm = 0
            rm = 0
            for j in singleP:
                
                start_time = time.time()
                accuracy = self.__fit(X[[j]], Y[[j]])
                end_time = time.time()
                rm = rm + end_time - start_time

                cm = (cm * j+accuracy)/(j+1)
                singleP.set_description("Epoch %2d" % (i+1))
                singleP.set_postfix_str("Accuracy: %.4f;" % (round(cm, 4)))

            epoch_accuracy[i] = cm
            epoch_time[i] = rm
        return [epoch_accuracy, epoch_time]

    def predict(self, X: np.ndarray) -> np.ndarray:
        for l in self.__layers:
            X = l.predict(X)
        return X