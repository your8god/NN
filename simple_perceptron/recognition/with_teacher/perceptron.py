import numpy as np


class Perceptron:
    """ Perceptron Rosenblatt """

    def __init__(self, n=0.01, round=10):
        self.n = n
        self.round = round

    def training(self, X: np.array, y) -> 'Perceptron':
        """ X: matrix; y: vector """

        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.round):
            errors = 0
            for xi, target in zip(X, y):
                out = self.predict(xi)
                d = target - out
                self.w[1:] += d * self.n * xi
                self.w[0] += d * self.n
                errors += d * self.n != 0.0
            self.errors.append(errors)
        
        return self
    
    def predict(self, X):
        """ Out-function """
        return np.where(
            (np.dot(X, self.w[1:]) + self.w[0]) >= 0.0,
            1,
            -1
        )


def res(val: int):
    if val == 1:
        print('Iris setosa')
    else:
        print('Iris versicolor')
