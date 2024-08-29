import numpy as np

from simple_neuron import Neuron


Xi = np.array([2, 3])
Wi = np.array([0, 1])
b = 4
neuron = Neuron(Wi, b)
print(f'Y = {neuron.y(Xi)}')