# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 19:13:15 2022

@author: smpsm
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

x = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
y = [1, 1, 1, -1, -1, 1, 1, 1]

n_input = 3   # 입력데이터의 변수 수
n_output = 1  # 출력데이터의 변수 수

perceptron = Sequential()
perceptron.add(Dense(units=n_output, activation='tanh',
                     input_shape=(n_input,),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros'))
'''
# 층 하나 더 추가
perceptron.add(Dense(units=n_output, activation='tanh',
                     input_shape=(n_input,),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros'))
'''
perceptron.compile(loss='mse', optimizer=SGD
                   (learning_rate=0.1), metrics=['mse'])
perceptron.fit(x, y, epochs=1500, verbose=2)

res=perceptron.predict(x)
print(res)

'''
# 층 하나일 때 출력결과
[[ 0.87848437]
 [ 0.99935544]
 [ 0.99935544]
 [-0.82511914]
 [ 0.9999968 ]
 [ 0.90113217]
 [ 0.90113217]
 [ 0.99948186]]

# 층 두 개일 때 출력 결과
[[0.49978504]
 [0.49948335]
 [0.49933568]
 [0.5009656 ]
 [0.49903378]
 [0.5006654 ]
 [0.50051844]
 [0.50021744]]

이 경우는 아마 과적합이 일어나서 오히려 결과가 더 안좋아진것 아닌가 싶다.
'''