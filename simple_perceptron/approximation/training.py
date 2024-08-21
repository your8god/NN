"""
X
  \
   w(k)
     \
       (N) -> Y
     /
   w(c)
  /
1 
"""

from random import uniform, choice


k, c = [uniform(-5, 5)] * 2
rate = 0.001 #speed of learning


def f(x):
    return x*k + c


def trainig(data: dict[float, float]) -> tuple[float, float]:
    """ Finding coef 'k' and 'c' for linear function (y = kx + c) using data"""
    global k, c
    for _ in range(100000): #training round
        x = choice(list(data.keys()))
        out = f(x)
        delta = data[x] - out #val of mistake
        k += delta * x * rate #w(t + 1) = w(t) + dxn
        c += delta * 1 * rate
    return k, c
