import numpy as np
from Functions.layers import Dense

class SGD:
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate
        self.t=0
    def update(self, layer):
        if isinstance(layer,Dense):
            layer.weights -= layer.dweights*self.learning_rate
            layer.bias -= layer.dbias*self.learning_rate


class SGDmomentum:
    def __init__(self,learning_rate, momentum=0.9):
        self.learning_rate=learning_rate
        self.momentum = momentum
        self.t=0
    def update(self, layer):
        if isinstance(layer,Dense):
            layer.m_w = (self.momentum * layer.m_w + (1 - self.momentum) * layer.dweights)
            layer.weights -= layer.m_w*self.learning_rate

            layer.m_b = (self.momentum * layer.m_b + (1 - self.momentum) * layer.dbias)
            layer.bias -= layer.m_b*self.learning_rate


class Adam:
    def __init__(self,learning_rate,beta1=0.9,beta2=0.999,eps=1e-8):
        self.learning_rate=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.t=0
        
    def update(self,layer):
        if isinstance(layer,Dense):
            layer.m_w = self.beta1*layer.m_w + (1-self.beta1)*layer.dweights
            layer.v_w = self.beta2*layer.v_w + (1-self.beta2)*(layer.dweights**2)
            m_w_hat = layer.m_w/(1-self.beta1**self.t)
            v_w_hat = layer.v_w/(1-self.beta2**self.t)
            
            layer.weights -= self.learning_rate*m_w_hat/(np.sqrt(v_w_hat)+self.eps)
            
            layer.m_b = self.beta1*layer.m_b + (1-self.beta1)*layer.dbias
            layer.v_b = self.beta2*layer.v_b + (1-self.beta2)*(layer.dbias**2)
            m_b_hat = layer.m_b/(1-self.beta1**self.t)
            v_b_hat = layer.v_b/(1-self.beta2**self.t)
            
            layer.bias -= self.learning_rate*m_b_hat/(np.sqrt(v_b_hat)+self.eps)
