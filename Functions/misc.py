import numpy as np
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