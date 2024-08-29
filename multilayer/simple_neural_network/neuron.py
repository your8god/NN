import numpy as np


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


class Neuron:
    """ Simple Neuron """

    def __init__(self, w: np.array, b: float):
        self.b = b
        self.w = w

    def y(self, x: np.array) -> float:
        res = np.dot(x, self.w) + self.b
        return sigmoid(res)
    