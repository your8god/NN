import numpy as np
from utils import load
from cnn import datasets


model = load()
X_train, y_train, X_test, y_test = datasets()
res = np.argmax(model.predict(X_test[:25]), axis=-1)
print(*res)
print(*[list(i).index(1) for i in y_test[:25]])
