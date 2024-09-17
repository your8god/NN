from sklearn.neighbors import KNeighborsClassifier

from ai_libraries.scikit_learn.dataset import datasets


X_train, X_test, y_train, y_test = datasets()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

res = knn.predict(X_test)
right = (y_test == res).sum()
wrong = len(res) - right
print(f'total: {len(res)}')
print(f'right: {right} ({round(right / len(res) * 100, 2)}%)')
print(f'wrong: {wrong} ({round(wrong / len(res) * 100, 2)}%)')
    