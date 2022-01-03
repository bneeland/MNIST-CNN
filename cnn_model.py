# Import libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

# Import data
(x_train_o, y_train_o), (x_test_o, y_test_o) = mnist.load_data()

# Reshape input data to specify the depth
x_train = x_train_o.reshape(x_train_o.shape[0], 28, 28, 1)
x_test = x_test_o.reshape(x_test_o.shape[0], 28, 28, 1)

# Convert data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# and normalize values
x_train /= 255
x_test /= 255
print(y_train_o)
print(y_test_o)
# Convert 1-dimensional class array to 10-dimensional class matrices
y_train = np_utils.to_categorical(y_train_o, 10)
y_test = np_utils.to_categorical(y_test_o, 10)
print(y_train)
print(y_test)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=7, batch_size=200)

# Save the model
model.save('saved_models/MNIST_CNN.h5')
