import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed = 1234)

X_train1 = np.array([[1,2,3,4]])
y_train1 = np.array([[4]])

model1 = Sequential()
model1.add(Dense(100, input_dim=4, kernel_initializer=initializer, use_bias=False))
model1.add(Dense(1, kernel_initializer=initializer, use_bias=False))
model1.compile(loss=MeanSquaredError(), optimizer=keras.optimizers.SGD(learning_rate=0.1))

#model1.layers[0].set_weights([np.array([[1,4],[2,5],[3,6]])])
#model1.layers[1].set_weights([np.array([[7],[8]])])

model1.fit(X_train1, y_train1, epochs=1)

print(model1.layers[0].get_weights()[0].T)
print(model1.layers[1].get_weights()[0].T)