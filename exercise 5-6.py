# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:42:47 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam,Adagrad,RMSprop
from sklearn.model_selection import KFold
import time
start = time.time() # 시간 측정을 위한 코드

# fashion MNIST 읽어와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)       # 텐서 모양 변환
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 신경망 구조 설정
n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

# 하이퍼 매개변수 설정
batch_siz = 256 # 한 번에 처리하는 샘플의 개수
n_epoch = 20    # 반복 횟수
k = 5 # 5-겹, 교차 검정 시 학습과 테스트 집합을 연속된 5개 폴더로 구분하여 사용

# 모델을 설계해주함는 수(모델을 나타내는 객체 modle을 반환)
def build_model(): # 같은 코드 4 번 반복하므로 함수 만들어 호출
    model = Sequential()
    model.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,)))
    model.add(Dense(units=n_hidden1, activation='relu'))
    model.add(Dense(units=n_hidden2, activation='relu'))
    model.add(Dense(units=n_hidden3, activation='relu'))
    model.add(Dense(units=n_hidden4, activation='relu'))
    model.add(Dense(units=n_output, activation='softmax'))
    return model

# 교차 검증을 해주는 함수(서로 다른 옵티마이저(opt)에 대해)
def cross_validation(opt):
    accuracy = []
    for train_index, val_index in KFold(k).split(x_train):
        xtrain, xval = x_train[train_index], x_train[val_index] # 인덱스의 훈련데이터와 검정데이터 값을 저장
        ytrain, yval = y_train[train_index], y_train[val_index]
        
        dmlp = build_model()
        dmlp.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        # 모델에서 오차는 categorical_crossentropy, accuracy는 행렬로 
        dmlp.fit(xtrain, ytrain, batch_size=batch_siz, epochs=n_epoch, verbose=0)
        accuracy.append(dmlp.evaluate(xval, yval, verbose=0)[1])
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

end = time.time()   # 시간 측정을 위한 코드
time = end - start  # 시간 측정을 위한 코드
print(time)