""" Simple neural network """

from sympy import exp, diff, symbols
import numpy as np


X = symbols('x')
F = 1 / (1 + exp(-X))


def sigmoid(x:float) -> float:
    return F.subs(X, x).n()


def sigmoid_diff(x: float) -> float:
    return diff(F).subs(X, x).n()


def mse(y_true: np.array, y: np.array) -> float:
    return ((y_true - y)**2).mean()


class Network:
    def __init__(self):
        for i in range(6):
            self.__dict__[f'w{i+1}'] = np.random.normal()
        for i in range(3):
            self.__dict__[f'b{i+1}'] = np.random.normal()
    
    def feedforward(self, x: np.array) -> int:
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o
    
    def trainig(self, data: np.array, all_y_trues: np.array):
        n = 1
        for i in range(1000):
            for x, y_true in zip(data, all_y_trues):
                # getting h1, h2, o
                s_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(s_h1)

                s_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(s_h2)

                s_o = self.w5 * x[0] + self.w6 * x[1] + self.b3
                o = sigmoid(s_o)
                y = o

                #diff
                dL_dy = -2 * (y_true - y)

                #Neuron o
                dy_dw5 = h1 * sigmoid_diff(s_o)
                dy_dw6 = h2 * sigmoid_diff(s_o)
                dy_db3 = sigmoid_diff(s_o)
                dy_dh1 = self.w5 * sigmoid_diff(s_o)
                dy_dh2 = self.w6 * sigmoid_diff(s_o)

                #Neuron h1
                dh1_dw1 = x[0] * sigmoid_diff(s_h1)
                dh1_dw2 = x[1] * sigmoid_diff(s_h1)
                dh1_db1 =  sigmoid_diff(s_h1)

                #Neuron h1
                dh2_dw3 = x[0] * sigmoid_diff(s_h2)
                dh2_dw4 = x[1] * sigmoid_diff(s_h2)
                dh2_db2 = sigmoid_diff(s_h2)

                #Update w's and b's
                #Neuron h1
                self.w1 -= n * dL_dy * dy_dh1 * dh1_dw1
                self.w2 -= n * dL_dy * dy_dh1 * dh1_dw2
                self.b1 -= n * dL_dy * dy_dh1 * dh1_db1

                #Neuron h2
                self.w3 -= n * dL_dy * dy_dh2 * dh2_dw3
                self.w4 -= n * dL_dy * dy_dh2 * dh2_dw4
                self.b2 -= n * dL_dy * dy_dh2 * dh2_db2

                #Neuron o
                self.w5 -= n * dL_dy  * dy_dw5
                self.w6 -= n * dL_dy  * dy_dw6
                self.b3 -= n * dL_dy  * dy_db3

            if i % 100 == 0:
                print(f"Iter: {i}; Loss: {mse(all_y_trues, y)}")
