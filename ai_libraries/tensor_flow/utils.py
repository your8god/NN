""" Utils for dataset """

import os
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path

from deep_learning import model


FASHION_KIND = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def dataset():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test


def transform(data):
    return data / 255.0


def visualisation(data, ind: int):
    plt.figure()
    plt.imshow(data[ind])
    plt.colorbar()
    plt.colorbar()
    plt.grid(False)
    plt.show()


def save(X, y):
    if not os.path.exists(_file()):
        my_model = model(X, y)
        my_model.save(_file())
    

def load():
    model = keras.models.load_model(_file())
    return model


def _file() -> Path:
    cur_path = (Path(__file__).parent / 'data').resolve()
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    return cur_path / 'model.h5'