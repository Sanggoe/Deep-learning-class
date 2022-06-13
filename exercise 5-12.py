# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:09:21 2022

@author: smpsm
"""
# exercise 5-12
'''
다층 퍼셉트론의 성능을 끌어올리는 방법

- 옵티마이저의 정확률을 비교했던 것을 토대로, optimizer는 adam 알고리즘을 사용하여 학습률을 자동으로 조절하도록 하여 학습시킨다.
- 배치 사이즈는 196으로, 전체 데이터 수에서 딱 나누어 떨어지는 적당한 크기로 안정성과 정확률을 높인다.
- 오차 함수는 오차값을 줄이기 위해 교차 엔트로피를 사용한다.
- 뉴런 수를 적절한 값으로 설정(1024개, 512개)
- 뉴런의 층 수를 2개의 층으로 축소하여 과적합을 방지한다.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

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
batch_siz = 196 # 한 번에 처리하는 샘플의 개수
n_epoch = 200    # 반복 횟수
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

# 옵티마이저에 대해 교차 검증을 실행
acc_adam = cross_validation(Adam())

# 정확률
print("Adam: ", np.array(acc_adam).mean())

import matplotlib.pyplot as plt

# 정확률 박스플롯
plt.boxplot([acc_adam], labels=["Adam"])
plt.grid()
plt.show()