# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:03:26 2022

@author: smpsm
"""

import numpy as np
import matplotlib.pyplot as plt

def stepfunc(x):
    return np.where(x < 0, -1, 1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (2/(1+np.exp(-x)))-1

def relu(x):
    return np.where(x <= 0, 0, x)


x = np.arange(-6, 6, 1)
y = stepfunc(x)

plt.plot(x, y)
plt.title('step function')
plt.show()

x = np.arange(-6, 6, 1)
y = sigmoid(x)

plt.plot(x, y)
plt.title('sigmoid function')
plt.show()

x = np.arange(-6, 6, 1)
y = tanh(x)

plt.plot(x, y)
plt.title('tanh function')
plt.show()

x = np.arange(-6, 6, 1)
y = relu(x)

plt.plot(x, y)
plt.title('relu function')
plt.show()