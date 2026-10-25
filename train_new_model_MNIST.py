import numpy as np
import pandas as pd
from Functions.preprocessing import numbers_to_one_hot
from Functions.losses import Cross_Entropy_Loss
from Functions.optimizers import SGD, SGDmomentum, Adam
from Functions.layers import Dense
from Functions.activations import ReLU, Softmax, Tanh, LeakyReLU
from Functions.train_test import Train, Test
from Functions.misc import save_model

train_dataset = pd.read_csv('train.csv')
train_sheet = np.array(train_dataset[1:]).T

train_data = train_sheet[1:].T/255
train_labels = numbers_to_one_hot(train_sheet[0])

model=[Dense(784,256),
       ReLU(),
       Dense(256,10),
       Softmax()]

learning_rate=0.001
Train(model, train_data, train_labels, batch_size=128, epochs=100, loss_func=Cross_Entropy_Loss, optimizer=Adam(learning_rate))

test_dataset = pd.read_csv('test.csv')
test_sheet = np.array(test_dataset[1:]).T

test_data = test_sheet[1:].T/255
test_labels = numbers_to_one_hot(test_sheet[0])

accuracy, errors = Test(model, test_data, test_labels, batch_size=5000, loss_func=Cross_Entropy_Loss)

#if accuracy[-1] > 0.9:
#    save_model(model, 'trained_model2.npz')