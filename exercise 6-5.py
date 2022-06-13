# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:59:04 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)  # 마지막 1은 확장성을 위해 사용, 컬러인 경우 3(rgb 값)으로
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

# relu 함수와 sigmoid 활성화 함수의 성능을 비교하는 문제
'''
relu 함수를 썼을 경우 정확률은 92.61000156402588

sigmoid 함수를 썼을 경우 정확률은 10.000000149011612 ???

sigmoid 함수를 사용했을 때 완전 정확률이 이상해졌다. relu가 더 좋다...??
'''
cnn = Sequential()
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(28, 28, 1)))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))


# 신경망 학습
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
hist=cnn.fit(x_train, y_train, batch_size=128, epochs=12,
             validation_data=(x_test, y_test), verbose=2)

res=cnn.evaluate(x_test, y_test, verbose=0)
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