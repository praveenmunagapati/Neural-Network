import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
class TanH:
    @staticmethod
    def activation(layer_input):
        return np.tanh(layer_input)

    @staticmethod
    def activation_prime(layer_input):
        return 1 - np.power(np.tanh(layer_input), 2)

class ReLU:
    @staticmethod
    def activation(layer_input):
        return np.where(layer_input >= 0, layer_input, 0)

    @staticmethod
    def activation_prime(layer_input):
        return np.where(layer_input >= 0, 1, 0)