# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:12:14 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# MNIST 읽어와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)       # 텐서 모양 변환
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

mlp = Sequential()
mlp.add(Dense(units=n_hidden1, activation='tanh', input_shape=(n_input,),
              kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden2, activation='tanh', input_shape=(n_input,),
              kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden3, activation='tanh', input_shape=(n_input,),
              kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden4, activation='tanh', input_shape=(n_input,),
              kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_output, activation='tanh',
              kernel_initializer='random_uniform', bias_initializer='zeros'))

# 신경망 학습
mlp.compile(loss='mean_squared_error', optimizer=Adam
                   (learning_rate=0.001), metrics=['accuracy'])
hist=mlp.fit(x_train, y_train, batch_size=128, epochs=30,
             validation_data=(x_test, y_test), verbose=2)

res=mlp.evaluate(x_test, y_test, verbose=0)
print("정확률은", res[1]*100)



import matplotlib.pyplot as plt

# 정확률 곡선
plt.plot(hist.history['accuracy'])     # train set에 대한 정확성 그래프
plt.plot(hist.history['val_accuracy']) # test set에 대한 정확성 그래프
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left') # 범례, loc(위치): 왼쪽 위
plt.grid()                                            # grid 그리기
plt.show()                                            # 그래프 그리기

# 손실 함수 곡선
plt.plot(hist.history['loss'])     # train set에 대한 오차 그래프
plt.plot(hist.history['val_loss']) # test set에 대한 오차 그래프
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show() 