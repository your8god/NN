from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn import metrics

from ai_libraries.scikit_learn.dataset import datasets

X_train, X_test, y_train, y_test = datasets()

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

p = Perceptron(eta0=0.1, random_state=0, max_iter=50)
p.fit(X_train_std, y_train)

res = p.predict(X_test_std)
right = (y_test == res).sum()
wrong = len(res) - right
print(f'total: {len(res)}')
print(f'right: {right} ({round(right / len(res) * 100, 2)}%)')
print(f'wrong: {wrong} ({round(wrong / len(res) * 100, 2)}%)')
print(metrics.accuracy_score(y_true=y_test, y_pred=res))