""" Convolutional neural nerwork """

from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def datasets():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def visualisation_of_data(X):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X[i], cmap=plt.cm.binary)
    plt.show()


def net_v1(X_train, y_train, X_test, y_test):
    model = Sequential()
    layer1 = Conv2D(64, kernel_size=3, activation='relu') #in
    layer2 = Conv2D(32, kernel_size=3, activation='relu')
    vec = Flatten() # converting 2D-vector to 1D-vector
    perceptorn = Dense(10, activation='softmax') #out
    for item in [layer1, layer2, vec, perceptorn]:
        model.add(item)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
    return model


def net_v2(X_train, y_train, X_test, y_test):
    model = Sequential()
    layer1 = Conv2D(64, kernel_size=3, activation='relu') #in
    layer2 = MaxPooling2D()
    layer3 = Conv2D(123, kernel_size=3, activation='relu')
    vec = Flatten() # converting 2D-vector to 1D-vector
    perceptorn = Dense(10, activation='softmax') #out
    for item in [layer1, layer2, layer3, vec, perceptorn]:
        model.add(item)
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
    return model
