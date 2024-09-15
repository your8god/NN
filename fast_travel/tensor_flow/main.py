import numpy as np

import utils


X_train, y_train, X_test, y_test = utils.dataset()
#utils.visualisation(X_train, 5)
X_train = utils.transform(X_train)
X_test = utils.transform(X_test)
#utils.visualisation(X_train, 5)

my_model = utils.load()
loss, acc = my_model.evaluate(X_test, y_test)
print(acc, loss, sep='\n')

res = my_model.predict(X_test)
right, wrong = 0, 0
for i, r in enumerate(res[:50]):
    print(
        'res:', 
        f'{utils.FASHION_KIND[np.argmax(r)]:<11}', 
        'total:', 
        f'{utils.FASHION_KIND[y_test[i]]:<11}',
        'predict:', 
        r.round(3)
    )
    right += utils.FASHION_KIND[np.argmax(r)] == utils.FASHION_KIND[y_test[i]]
    wrong += utils.FASHION_KIND[np.argmax(r)] != utils.FASHION_KIND[y_test[i]]
print('right', right, 'wrong', wrong)
