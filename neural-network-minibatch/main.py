'''Written by Nicholas Passantino 6/2018'''

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
import os

# get dataset
digits = datasets.load_digits()
x = digits.data
y1 = digits.target
y = []

# replace y values with lists that have a 1 at the index matching the original value, for ex: 3 -> [0,0,0,1,0,0,0,0,0,0]
# this way classification and error calculation can be done easily
for item in y1:
    element = [0] * 10
    element[item] = 1
    y.append(element)

# first half of dataset is for training, second half for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

nn = NeuralNetwork(64, 37, 10)

# if all storage files exist and are not empty, load weights and biases
if os.path.exists('wh.txt') and os.path.getsize('wh.txt') > 0\
        and os.path.exists('bh.txt') and os.path.getsize('bh.txt') > 0\
        and os.path.exists('wo.txt') and os.path.getsize('wo.txt') > 0\
        and os.path.exists('bo.txt') and os.path.getsize('bo.txt') > 0:

    nn.load_weights_biases()

# else must train the network
else:
    nn.train(x_train, y_train, 64)

num_right = 0
for i in range(0, len(y_test)):

    predict = nn.predict(x_test[i])
    actual = np.argmax(y_test[i])

    if predict == actual:
        num_right += 1
print(str(num_right) + ' out of ' + str(len(y_test)) + ' were correct: ' + str(num_right / len(y_test)) + '% success')

