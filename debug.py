import NeuralNetwork as nn
import numpy as np
from tensorflow import keras

initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed = 1234)

m = nn.model()
m.addLayer(nn.InputLayer(3))
m.addLayer(nn.Layer(3, 4).initializeWeight(initializer))
m.addLayer(nn.Layer(4, 5).initializeWeight(initializer))
m.addLayer(nn.Layer(5, 2).initializeWeight(initializer))

X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
Y = np.array([[23,24], [25,26],[27,28],[29,30]])
m.fit0(X, Y, 0.1)

print(m.predict(np.array([[13,14,15]])))