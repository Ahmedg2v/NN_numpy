import numpy as np

def numbers_to_one_hot(numbers, num_classes=10):
    # Initialize an array of zeros
    one_hot_vectors = np.zeros((len(numbers), num_classes))
    
    # Set the appropriate index to 1 for each number
    for idx, number in enumerate(numbers):
        one_hot_vectors[idx, number] = 1
    
    return one_hot_vectors

def minmax_norm(x,a,b):
    x_new = a + (x-np.min(x,axis=1, keepdims=True))*(b-a)/(np.max(x,axis=1, keepdims=True)-np.min(x,axis=1, keepdims=True))
    return x_new

def tanh_norm(x):
    x_new = np.tanh(x/np.max(np.abs(x), axis=1, keepdims=True))
    return x_new
