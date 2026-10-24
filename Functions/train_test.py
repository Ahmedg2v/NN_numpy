import numpy as np
from Functions.batches import Batch_gen

def Train(model, train_data, train_labels, batch_size, epochs, loss_func, optimizer):
    errors=[]
    accuracy=[]

    for epoch in range(epochs):
        batch = Batch_gen(train_data,train_labels,batch_size)
        output = batch[0]
        real = batch[1]
        
        for layer in model:
            output = layer.forward(output)
        
        accuracy.append(np.sum(np.argmax(output, axis=1) ==  np.argmax(real, axis=1)) / len(np.argmax(output, axis=1)))  
        errors.append(loss_func.forward(output,real))

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Accuracy: {accuracy[-1]}, Loss: {errors[-1]}')
        
        grad = loss_func.backward(output, real)
    
        for layer in reversed(model[:-1]):
            grad = layer.backward(grad)

        optimizer.t += 1
        for layer in model:
            optimizer.update(layer)

    return np.array(accuracy), np.array(errors)

def Test(model, test_data, test_labels, batch_size, loss_func):
    errors=[]
    accuracy=[]

    batch = Batch_gen(test_data,test_labels,batch_size)
    output = batch[0]
    real = batch[1]
    
    for layer in model:
        output = layer.forward(output)
    
    accuracy.append(np.sum(np.argmax(output, axis=1) ==  np.argmax(real, axis=1)) / len(np.argmax(output, axis=1)))  
    errors.append(loss_func.forward(output,real))
    print(f'Test - Accuracy: {accuracy[-1]}, Loss: {errors[-1]}')
    return np.array(accuracy), np.array(errors)