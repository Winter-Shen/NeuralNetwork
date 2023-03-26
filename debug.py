import NeuralNetwork as nn
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import SGD

X_train = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y_train = np.array([[23,24], [25,26],[27,28],[29,30]])
X_test = np.array([[13,14,15]])

'''
My Network
'''
initializer = RandomNormal(mean=0., stddev=1., seed = 12345)

m1 = nn.MyModel()
m1.addLayer(nn.InputLayer(3))

#m.addLayer(nn.MyLayer(3, 4).initializeWeight(initializer))
#m.addLayer(nn.MyLayer(4, 5).initializeWeight(initializer))
#m.addLayer(nn.MyLayer(5, 2).initializeWeight(initializer))

m1.addLayer(nn.MyLayer(3, 4).setWeight(np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])))
m1.addLayer(nn.MyLayer(4, 5).setWeight(np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])))
m1.addLayer(nn.MyLayer(5, 2).setWeight(np.array([[37,38],[39,40],[41,42],[43,44],[45,46]])))

#m.selectNodes([[1,3],[0,2,4]])
m1.fit(X_train, y_train, 0.1, epochs=1)

y_hat = m1.predict(X_test)
print(y_hat)

'''
Keras Network
'''

initializer = RandomNormal(mean=0., stddev=1., seed = 12345)

m2 = Sequential()
m2.add(Dense(4, input_dim=3, kernel_initializer=initializer, use_bias=False))
m2.add(Dense(5, kernel_initializer=initializer, use_bias=False))
m2.add(Dense(2, kernel_initializer=initializer, use_bias=False))
m2.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.1))

m2.layers[0].set_weights([np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])])
m2.layers[1].set_weights([np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])])
m2.layers[2].set_weights([np.array([[37,38],[39,40],[41,42],[43,44],[45,46]])])

m2.fit(X_train, y_train, epochs=1)

p = m2.predict(X_test)
print(p)