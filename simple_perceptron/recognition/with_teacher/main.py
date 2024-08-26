import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron, res



_TO_DOWNLOAD = False


if _TO_DOWNLOAD:
    import download_dataset

with open('X.npy', 'rb') as Xi, \
    open('y.npy', 'rb') as yi:
    X = np.load(Xi)
    y = np.load(yi)

prn = Perceptron(n=0.1)
prn.training(X, y)
plt.plot(range(1, len(prn.errors) + 1), prn.errors, marker='o')
plt.xlabel('Rounds')
plt.ylabel('Errors')
plt.show()

r1 = prn.predict([5.5, 1.6])
r2 = prn.predict([6.4, 4.5])
res(r1)
res(r2)