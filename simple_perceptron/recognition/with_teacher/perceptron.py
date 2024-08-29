import numpy as np
from abc import abstractmethod, ABC


class Perceptron(ABC):
    """ Base Perceptron """

    def __init__(self, n=0.01, round=10):
        self.n = n
        self.round = round

    @abstractmethod
    def training(self, X: np.array, y) -> 'Perceptron':
        raise NotImplementedError
    
    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """ Out-function """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

def res(val: int):
    if val == 1:
        print('Iris setosa')
    else:
        print('Iris versicolor')
        