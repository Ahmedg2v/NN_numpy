import numpy as np

class Cross_Entropy_Loss:
    def forward(output,real):
        entropy= -np.sum(real*np.log(np.clip(output, 1e-12, 1.0)))/len(output)
        return entropy
    
    def backward(output,real):
        grad=(output - real) / len(output)
        return grad
