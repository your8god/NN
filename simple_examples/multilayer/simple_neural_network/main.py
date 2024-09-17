import numpy as np

from multilayer_net import NeuralNetwork
from neuron import Neuron


Xi = np.array([2, 3])
Wi = np.array([0, 1])
b = 4
neuron = Neuron(Wi, b)
print(f'Y = {neuron.y(Xi)}')

nn = NeuralNetwork(np.array([0, 1]), 0)
x = np.array([2, 3])
print(f'Y = {nn.feedforward(x)}')