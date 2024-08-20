from random import randint

import simple_perceptron


def training(topic: int, n: int):
    """ Training of perceptron """
    for _ in range(n):
        val = randint(0, 9)
        res = simple_perceptron.perceptron(simple_perceptron.NUMBERS[val])

        if val == topic:
            if not res:
                simple_perceptron.increase(simple_perceptron.NUMBERS[val])
        else:
            if res:
                simple_perceptron.decrease(simple_perceptron.NUMBERS[val])
