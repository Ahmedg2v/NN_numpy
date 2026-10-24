import numpy as np
from Functions.base import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.random.randn(output_size,1)
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        
    def forward(self, input):
        self.input = input.reshape(input.shape[0], -1)
        self.output = self.input @ self.weights.T + self.bias.T
        return self.output

    def backward(self, grad):
        self.dweights = grad.T@self.input
        self.dbias = np.sum(grad, axis=0, keepdims=True).T
        grad = grad@self.weights
        return grad
