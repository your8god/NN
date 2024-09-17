""" Loading of dataset """

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def datasets():
    X = load_iris().data
    y = load_iris().target
    return train_test_split(X, y, random_state=0)
