from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


class model:
    def __init__(self, x, y, layer):
        self.layersNumber = len(layer)
        self.layer = layer

        self.data = [None for i in layer]
        self.data[0] = np.matrix(x).T

        self.trueData = [None for i in range(self.layersNumber)]
        self.trueData[(self.layersNumber)-1] = np.matrix(y).T

        self.weight = [[None for j in range(layer[i+1])] for i in range(self.layersNumber - 1)]

    def gweight(self, initializer):
        for i in range(self.layersNumber - 1):
            matrix = initializer(shape = (self.layer[i], self.layer[i+1])).numpy().T
            for j, row in enumerate(matrix):
                self.weight[i][j] = np.matrix(row)
    
    def forward(self):
        for i, weight in enumerate(self.weight):
            data=[]
            for j, w in enumerate(weight):
                data.append((w * self.data[i])[0,0])
            self.data[i+1] = np.matrix(data).T
        diff = self.data[self.layersNumber-1] - self.trueData[(self.layersNumber)-1]
        self.Loss = diff * (diff.T)

    def backward(self, learning_rate):
        d = 2 * (self.data[self.layersNumber-1] - self.trueData[self.layersNumber-1])
        for i in range(self.layersNumber-1, 0 ,-1):
            if(i == 2):
                d1 = d * self.weight[1][0]
            else:
                d1 = None
            for j, w in enumerate(self.weight[i-1]):                
                self.weight[i-1][j]  = w - learning_rate * d[0, j] *(self.data[i-1].T)
            d = d1

m = model([1,2,3,4], 4, [4, 100, 1])
initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed = 1234)
#m.weight = [[np.matrix([1,2,3]), np.matrix([4,5,6])],[np.matrix([7,8])]]
m.gweight(initializer)
m.forward()
m.backward(0.1)

print(np.array([i.tolist()[0] for i in m.weight[0]]))
print(np.array([i.tolist()[0] for i in m.weight[1]]))