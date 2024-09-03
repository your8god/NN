import numpy as np

from net import Network


def normalize(arr):
    means = np.mean(arr, axis=0)
    return arr - means


#weight and heigh of person
data = np.array([
    [133, 65],
    [160, 72],
    [152, 70],
    [120, 60.0]
])

#1 - man, 2 - woman
all_y_trues = np.array([1, 0, 0, 1])

data = normalize(data)

network = Network()
network.trainig(data, all_y_trues)
print(*[network.feedforward(i) for i in data])