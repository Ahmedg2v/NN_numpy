import numpy as np
import pandas as pd
from sympy import false
from Functions.preproccessing import numbers_to_one_hot
from Functions.losses import Cross_Entropy_Loss
from Functions.optimizers import SGD, SGDmomentum, Adam
from Functions.layers import Dense
from Functions.activations import ReLU, Softmax, Tanh, LeakyReLU
from Functions.train_test import Train, Test
import gc

params = np.load('trained_model1.npz', allow_pickle=True)

model=[Dense(784,256),
       ReLU(),
       Dense(256,10),
       Softmax()]

model[0].weights=params['w1']
model[0].bias=params['b1']
model[2].weights=params['w2']
model[2].bias=params['b2']

test_dataset = pd.read_csv('test.csv')
test_sheet = np.array(test_dataset[1:]).T

test_data = test_sheet[1:].T/255
test_labels = numbers_to_one_hot(test_sheet[0])

accuracy, errors = Test(model, test_data, test_labels, batch_size=5000, loss_func=Cross_Entropy_Loss)

# use Train() to further train the model if needed:
retrain = False
if retrain:
    train_dataset = pd.read_csv('train.csv')
    train_sheet = np.array(train_dataset[1:]).T

    train_data = train_sheet[1:].T/255
    train_labels = numbers_to_one_hot(train_sheet[0])

    learning_rate=0.001

    Train(model, train_data, train_labels, batch_size=128, epochs=100, loss_func=Cross_Entropy_Loss, optimizer=Adam(learning_rate))

       
gc.collect()