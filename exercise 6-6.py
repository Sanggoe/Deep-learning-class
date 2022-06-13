# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:01:06 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import SGD,Adam,Adagrad,RMSprop
from sklearn.model_selection import KFold

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)  # 마지막 1은 확장성을 위해 사용, 컬러인 경우 3(rgb 값)으로
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)


# 하이퍼 매개변수 설정
batch_siz = 128 # 한 번에 처리하는 샘플의 개수
n_epoch = 12    # 반복 횟수
k = 5 # 5-겹, 교차 검정 시 학습과 테스트 집합을 연속된 5개 폴더로 구분하여 사용


def build_model(): # 같은 코드 4 번 반복하므로 함수 만들어 호출
    cnn = Sequential()
    cnn = Sequential()
    cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(Conv2D(64,(3,3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(10, activation='softmax'))
    return cnn


# 교차 검증을 해주는 함수(서로 다른 옵티마이저(opt)에 대해)
def cross_validation(opt):
    accuracy = []
    for train_index, val_index in KFold(k).split(x_train):
        xtrain, xval = x_train[train_index], x_train[val_index] # 인덱스의 훈련데이터와 검정데이터 값을 저장
        ytrain, yval = y_train[train_index], y_train[val_index]
        
        cnn = build_model()
        
        # 신경망 학습
        cnn.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        cnn.fit(xtrain, ytrain, batch_size=batch_siz, epochs=n_epoch, verbose=2)
        accuracy.append(cnn.evaluate(xval, yval, verbose=0)[1])
    return accuracy # accuracy 행렬의 두번째 원소로 검증집합에 대한 정확성을 추가, 첫번째 원소는 인덱스


# 옵티마이저 4개에 대해 교차 검증을 실행
acc_sgd = cross_validation(SGD())
acc_adam = cross_validation(Adam())
acc_adagrad = cross_validation(Adagrad())
acc_rmsprop = cross_validation(RMSprop())


# 옵티마이저 4개의 정확률 비교
print("SGD: ", np.array(acc_sgd).mean()) # 각 폴더에서 SGD의 정확성 평균 출력
print("Adam: ", np.array(acc_sgd).mean())
print("Adagrad: ", np.array(acc_sgd).mean())
print("RMSprop: ", np.array(acc_sgd).mean())

import matplotlib.pyplot as plt

# 네 옵티마이저의 정확률을 박스플롯으로 비교
plt.boxplot([acc_sgd, acc_adam, acc_adagrad, acc_rmsprop], labels=["SGD", "Adam", "Adagrad", "RMSprop"])
plt.grid()
plt.show()