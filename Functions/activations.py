import numpy as np
from Functions.base import Layer

class Activation(Layer):
    def __init__(self, activation, activation_drv):
        self.activation = activation
        self.activation_drv = activation_drv
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, grad):
        grad = grad*self.activation_drv(self.input)
        return grad

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        
        def relu_drv(x):
            return (x > 0)*1

        super().__init__(relu, relu_drv)
        
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_drv(x):
            return 1-(np.tanh(x)**2)
        
        super().__init__(tanh, tanh_drv)

class LeakyReLU(Activation):
    def __init__(self, alpha):
        def leakyrelu(x):
            return (x > 0)*x + (x <= 0)*alpha*x
        
        def leakyrelu_drv(x):
            return (x > 0)*1 + (x <= 0)*alpha

        super().__init__(leakyrelu, leakyrelu_drv)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = tmp/np.sum(tmp,axis=1,keepdims=True)
        return self.output
