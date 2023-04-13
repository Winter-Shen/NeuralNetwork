import NeuralNetwork as nn
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.initializers import RandomNormal
from keras.optimizers import SGD

X_train = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y_train = np.array([[23], [25],[27],[29]])
X_test = np.array([[13,14,15]])
w1 = np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])
w2 = np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])
w3 = np.array([[37,38],[39,40],[41,42],[43,44],[45,46]])

'''
My Network
'''
initializer = RandomNormal(mean=0., stddev=1., seed = 12345)

m1 = nn.MyModel()
m1.addLayer(nn.InputLayer(3))

m1.addLayer(nn.MyLayer(3, 4).setWeight(w1))
m1.addLayer(nn.MyLayer(4, 5).setWeight(w2))
m1.addLayer(nn.MyLayer(5, 2).setWeight(w3))

m1.fit(X_train, y_train, 0.1, epoches=1, batch_size = 4)

#y_hat = m1.predict(X_test)
#print(y_hat)

'''
Keras Network
'''

X_train = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y_train = np.array([[23], [25],[27],[29]])
X_test = np.array([[13,14,15]])
w1 = np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])
w2 = np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])
w3 = np.array([[37,38],[39,40],[41,42],[43,44],[45,46]])

initializer = RandomNormal(mean=0., stddev=1., seed = 12345)

m2 = Sequential()
m2.add(Dense(4, input_dim=3, use_bias=False))
m2.add(Dense(5, use_bias=False))
m2.add(Dense(2, use_bias=False))
m2.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])

m2.layers[0].set_weights([w1])
m2.layers[1].set_weights([w2])
m2.layers[2].set_weights([w3])
m2.fit(X_train, y_train, epochs=1, batch_size=4)

#y_hat1 = m2.predict(X_test)
#print(y_hat1)