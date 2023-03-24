import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError


initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed = 1234)

X_train1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y_train1 = np.array([[23], [24],[25],[26]])

model1 = Sequential()
model1.add(Dense(4, input_dim=3, kernel_initializer=initializer, use_bias=False))
model1.add(Dense(5, kernel_initializer=initializer, use_bias=False))
model1.add(Dense(1, kernel_initializer=initializer, use_bias=False))
model1.compile(loss=MeanSquaredError(), optimizer=keras.optimizers.SGD(learning_rate=0.1))

model1.layers[0].set_weights([np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])])
model1.layers[1].set_weights([np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])])
model1.layers[2].set_weights([np.array([[37],[38],[39],[40],[41]])])

model1.fit(X_train1, y_train1, epochs=1)

#print(model1.layers[0].get_weights()[0].T)
#print(model1.layers[1].get_weights()[0].T)
p = model1.predict(np.array([[13,14,15]]))
print(p)