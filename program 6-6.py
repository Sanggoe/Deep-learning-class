# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:23:37 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()
# C C P  C C P  FC FC
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(Conv2D(32,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

# 신경망 학습
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
hist=cnn.fit(x_train, y_train, batch_size=128, epochs=12,
             validation_data=(x_test, y_test), verbose=2)

cnn.summary()   # cnn 모델의 정보 출력

### 커널의 시각화
for layer in cnn.layers:
    if 'conv' in layer.name:
        kernel, biases = layer.get_weights()
        print(layer.name, kernel.shape) # 커널의 텐서 모양을 출력
        
kernel, biases = cnn.layers[0].get_weights()    # 층 0의 커널 정보를 저장
minv, maxv = kernel.min(), kernel.max()
kernel = (kernel-minv)/(maxv-minv)
n_kernel = 32

import matplotlib.pyplot as plt

plt.figure(figsize=(20,3))
plt.suptitle("Kernels of conv2d_4")
for i in range(n_kernel):   # i 번째 커널
    f = kernel[:,:,:,i]     # kernel[:,:,:,i]: 첫 세 개의 모든 인덱스와 네번째 인덱스 중 i값을 가지는 원소 
    for j in range(3):      # j 번째 커널
        plt.subplot(3, n_kernel, j*n_kernel+i+1)
        plt.imshow(f[:,:,j], cmap='gray')   # f[:, :, j]의 값으로 2차원 컬러행렬 생성
        plt.xticks([]); plt.yticks([])  # X축과 Y축의 값 표현
        plt.title(str(i) + '_' + str(j))
plt.show()
    
### 특징맵 시각화
for layer in cnn.layers:
    if 'conv' in layer.name:
        print(layer.name, layer.output.shape)
        
from tensorflow.keras.models import Model

partial_model = Model(inputs = cnn.inputs, outputs = cnn.layers[0].output)
partial_model.summary()

feature_map = partial_model.predict(x_test)
fm = feature_map[1]

plt.imshow(x_test[1])

plt.figure(figsize=(20,3))
plt.suptitle("Feature maps of conv2d_4")
for i in range(32):
    plt.subplot(2, 16, i+1)
    plt.imshow(fm[:,:,i], cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title("map"+str(i))
plt.show()