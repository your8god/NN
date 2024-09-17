""" Functions for learning the simple perceptron """

NUMBERS = [
    '111101101101111', #0
    '001001001001001', #1
    '111001111100111', #2
    '111001111001111', #3
    '101101111001001', #4
    '111100111001111', #5
    '111100111101111', #6
    '111001001001001', #7
    '111101111101111', #8
    '111101111001111', #9
]

w = [0] * 15 #a digit is a grid 3x5

def perceptron(sensor: str) -> int:
    """ Return result """
    s = 0
    b = 7 #limit

    for i in range(15):
        s += w[i] * int(sensor[i])
    return s >= b


def increase(number: str):
    """ Mistake of type 1 """
    for i in range(15):
        if int(number[i]):
            w[i] += 1


def decrease(number: str):
    """ Mistake of type 2 """
    for i in range(15):
        if int(number[i]):
            w[i] -= 1
