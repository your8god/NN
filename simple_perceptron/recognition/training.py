from random import randint

import perceptron as p


def training(topic: int, n: int):
    """ Training of perceptron """
    for _ in range(n):
        val = randint(0, 9)
        res = p.perceptron(p.NUMBERS[val])

        if val == topic:
            if not res:
                p.increase(p.NUMBERS[val])
        else:
            if res:
                p.decrease(p.NUMBERS[val])
