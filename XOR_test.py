import numpy as np
from Functions.losses import MSE
from Functions.optimizers import Adam
from Functions.layers import Dense
from Functions.activations import ReLU, Sigmoid
from Functions.misc import save_model
from Functions.batches import Batch_gen, Batch_shuffle

def Train_xor(model, train_data, train_labels, epochs, loss_func, optimizer):
    errors=[]
    accuracy=[]

    for epoch in range(epochs):
        batch = Batch_shuffle(train_data,train_labels)
        output = batch[0]
        real = batch[1]
        
        for layer in model:
            output = layer.forward(output)

        predictions = np.where(output >= 0.5, 1, 0)
        accuracy.append(np.sum(predictions == real) / len(real))
        errors.append(loss_func.forward(output,real))

        #if epoch % 10 == 0:
         #   print(f'Epoch: {epoch}, Accuracy: {accuracy[-1]}, Loss: {errors[-1]}')
        
        grad = loss_func.backward(output, real)
    
        for layer in reversed(model):
            grad = layer.backward(grad)

        optimizer.t += 1
        for layer in model:
            optimizer.update(layer)

    return np.array(accuracy), np.array(errors)

def Test_xor(model, test_data, test_labels, batch_size, loss_func):
    errors=[]
    accuracy=[]

    batch = Batch_gen(test_data,test_labels,batch_size)
    output = batch[0]
    real = batch[1]
    
    for layer in model:
        output = layer.forward(output)

    predictions = np.where(output >= 0.5, 1, 0)
    accuracy.append(np.sum(predictions == real) / len(real))
    errors.append(loss_func.forward(output,real))

    print(f'Test - Accuracy: {accuracy[-1]}, Loss: {errors[-1]}')

    return np.array(accuracy), np.array(errors)


truth_table = np.array([[0, 0, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]])

train_data = truth_table[:, :2]
train_labels = truth_table[:, 2].reshape(-1,1)

model = [Dense(2, 8),
         ReLU(),
         Dense(8, 1),
         Sigmoid()]

learning_rate=0.01
Train_xor(model, train_data, train_labels, epochs=1000, loss_func=MSE, optimizer=Adam(learning_rate))

Test_xor(model, train_data, train_labels, batch_size=1000, loss_func=MSE)

#if accuracy==1:
 #   save_model(model,'XOR_model.npz')
