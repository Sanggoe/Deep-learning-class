# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:34:00 2022

@author: smpsm
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# OR 데이터 구축
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
y = [[-1], [1], [1], [1]]

n_input = 2   # 입력데이터의 변수 수
n_output = 1  # 출력데이터의 변수 수

perceptron = Sequential()
perceptron.add(Dense(units=n_output, activation='tanh',
                     input_shape=(n_input,),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros'))

perceptron.compile(loss='mse', optimizer=SGD
                   (learning_rate=0.1), metrics=['mse'])
perceptron.fit(x, y, epochs=500, verbose=2)

res=perceptron.predict(x)
print(res)