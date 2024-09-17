r"""
NeuralNetwork Scheme

1-layer  2-l 3-l

X1 = 2  - H1  
       \ /  \
        X    OUT -> Y
       / \  /
X2 = 3  - H2
"""

import numpy as np

from neuron import Neuron


class NeuralNetwork:
    """ 3-layers nn """

    def __init__(self, w: np.array, b: float):
        self.w = w
        self.b = b

        self.inside1 = Neuron(w, b)
        self.inside2 = Neuron(w, b)
        self.output = Neuron(w, b)

    def feedforward(self, x: np.array) -> float:
        out_inside1 = self.inside1.y(x)
        out_inside2 = self.inside2.y(x)
        out = self.output.y(np.array([out_inside1, out_inside2]))
        return out
    