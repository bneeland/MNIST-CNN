# Import libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
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

# Convert 1-dimensional class array to 10-dimensional class matrices
y_train = np_utils.to_categorical(y_train_o, 10)
y_test = np_utils.to_categorical(y_test_o, 10)

# Retrieve the model
from keras.models import load_model
model = load_model('saved_models/MNIST_CNN.h5')

# Predict an example from the test data
x_prediction = model.predict(x_test)
i = np.random.randint(low=0, high=9999)
print("Sample number:", i)
print("Label:", y_test_o[i])
print("Prediction:", x_prediction[i])
plt.imshow(x_test_o[i])
plt.show()
