import numpy as np

from perceptron import Perceptron


class AdaptiveLinearNeuron(Perceptron):
    """ ADALINE """

    def training(self, X, y) -> 'AdaptiveLinearNeuron':
        self.w = np.zeros(1 + X.shape[1])
        self.cost = []

        for _ in range(self.round):
            out = self.net_input(X)
            d = y - out
            self.w[1:] += self.n * X.T.dot(d)
            self.w[0] += self.n * d.sum()
            cost = (d**2).sum() / 2.0
            self.cost.append(cost)
        
        return self
    