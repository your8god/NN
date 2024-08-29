import numpy as np

from perceptron import Perceptron


class PerceptronRosebblatt(Perceptron):
    """ Perceptron Rosenblatt """

    def training(self, X: np.array, y) -> 'PerceptronRosebblatt':
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
