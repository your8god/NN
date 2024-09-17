from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from fast_travel.scikit_learn.dataset import datasets


X_train, X_test, y_train, y_test = datasets()

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)

res = lr.predict_proba(X_test)
for row, ans_row in zip(res, y_test):
    zero, one, two = (row * 100).round(2)
    d = {0: zero, 1:one, 2:two}
    res_net = max(d, key=lambda x: d[x])
    print(f'0: {zero:<5} 1: {one:<5} 2: {two:<5} ans: {ans_row} net_res: {res_net} \
verdict: {"success" if res_net == ans_row else "failure"}')
