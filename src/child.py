# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)

# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape)
print(x_test.shape)

print(x_train[0])
print(y_train[0])

# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])

# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

# Build the model architecture with Parent APP
model = Sequential()

model.add(Dense(1024, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# TODO: Compile the model using a loss function and an optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])

model.summary()

# TODO: Run the model. Feel free to experiment with different batch sizes and number of epochs.
model.fit(x_test, y_test, epochs=7, batch_size=500, verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: ", score[1])
