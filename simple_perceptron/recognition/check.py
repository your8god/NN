from training import training
from perceptron import w, perceptron, NUMBERS


n = 10000
topic = 5
training(topic, n)
print(w)

print("0 is 5?", perceptron(NUMBERS[0]))
print("1 is 5?", perceptron(NUMBERS[1]))
print("2 is 5?", perceptron(NUMBERS[2]))
print("3 is 5?", perceptron(NUMBERS[3]))
print("4 is 5?", perceptron(NUMBERS[4]))
print("5 is 5?", perceptron(NUMBERS[5]))
print("6 is 5?", perceptron(NUMBERS[6]))
print("7 is 5?", perceptron(NUMBERS[7]))
print("8 is 5?", perceptron(NUMBERS[8]))
print("9 is 5?", perceptron(NUMBERS[9]))
print()

almost_rs = [
    '111100111000111',
    '111100010001111',
    '111100011001111',
    '110100111001111',
    '110100111001011',
    '111100101001111',
]
print('is 5 in 51?', perceptron(almost_rs[0]))
print('is 5 in 52?', perceptron(almost_rs[1]))
print('is 5 in 53?', perceptron(almost_rs[2]))
print('is 5 in 54?', perceptron(almost_rs[3]))
print('is 5 in 55?', perceptron(almost_rs[4]))
print('is 5 in 56?', perceptron(almost_rs[5]))