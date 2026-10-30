import numpy as np
import matplotlib.pyplot as plt
from Functions.layers import Dense

def save_model(model, filename):
    architecture = [f'Dense({layer.input_size}, {layer.output_size})' if isinstance(layer, Dense) else type(layer).__name__ for layer in model]

    params = {}
    dense_counter = 1
    for layer in model:
        if isinstance(layer, Dense):
            params[f"w{dense_counter}"] = layer.weights
            params[f"b{dense_counter}"] = layer.bias
            dense_counter += 1
    
    np.savez(filename, **params, architecture=architecture)

def plot(x,caption):
    data=x
    t = np.arange(0,len(x))
    fig = plt.figure(figsize=(10,6))
    
    plt.plot(t, data, 'g')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'Accuracy (%)', fontsize=14)
    plt.title(f'{caption}', fontsize=14)
    plt.show()
