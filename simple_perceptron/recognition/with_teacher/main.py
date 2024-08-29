import numpy as np
import matplotlib.pyplot as plt

from rosenblatt import PerceptronRosebblatt
from adaline import AdaptiveLinearNeuron
from perceptron import res


_TO_DOWNLOAD = False


def normalization(data, i):
    return (data - X[:, i].mean()) / X[:, i].std()


if _TO_DOWNLOAD:
    import download_dataset

with open('X.npy', 'rb') as Xi, \
    open('y.npy', 'rb') as yi:
    X = np.load(Xi)
    y = np.load(yi)

prn = PerceptronRosebblatt(n=0.1)
prn.training(X, y)
plt.plot(range(1, len(prn.errors) + 1), prn.errors, marker='o')
plt.xlabel('Rounds')
plt.ylabel('Errors')
plt.show()

r1 = prn.predict([5.5, 1.6])
r2 = prn.predict([6.4, 4.5])
res(r1)
res(r2)

aln = AdaptiveLinearNeuron()
X_std = np.copy(X)
# normalization
X_std[:, 0] = normalization(X[:, 0], 0)
X_std[:, 1] = normalization(X[:, 1], 1)
aln.training(X_std, y)
plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel('Rounds')
plt.ylabel('Residual sum of squares')
plt.show()

arr1 = [5.5, 1.6]
arr1[0] = normalization(arr1[0], 0)
arr1[1] = normalization(arr1[1], 1)
print(arr1)
r1 = aln.predict(arr1)
arr2 = [6.4, 4.5]
arr2[0] = normalization(arr2[0], 0)
arr2[1] = normalization(arr2[1], 1)
print(arr2)
r2 = aln.predict(arr2)
res(r1)
res(r2)
