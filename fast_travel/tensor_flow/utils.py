""" Utils for dataset """

from tensorflow import keras
import matplotlib.pyplot as plt

from deep_learning import model


_FILE = 'model_1.h5'


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
    my_model = model(X, y)
    my_model.save(_FILE)
    

def load():
    model = keras.models.load_model(_FILE)
    return model
