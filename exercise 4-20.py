# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:55:11 2022

@author: smpsm
"""

# 이거 아닌거 같은데 ㅋㅋㅋㅋ 이거 나오면 큰일 나는거다!!

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

# training set 구축
X = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
Y = [1, 1, 1, -1, -1, 1, 1, 1]

''' 모델 객체 생성 '''
mlp = MLPClassifier(hidden_layer_sizes=3, learning_rate_init=0.001,
                    batch_size=8, max_iter=10, solver='adam', verbose=True)
''' 모델 학습 ''' 
mlp.fit(X, Y) # 학습

''' 학습된 모델로 예측 '''
res=mlp.predict(X) # 테스트 집합으로 예측

''' 성능 측정 '''
# 혼동 행렬 구함
conf=np.zeros((8,8)) 
for i in range(len(res)):
    conf[res[i]][Y[i]] += 1
print(conf)

# 정확률 계산
no_correct = 0
for i in range(8):
    no_correct += conf[i][i]
accuracy = no_correct/len(res)
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")



뭘까 ㅋㅋㅋㅋ


import tensorflow as tf

# training set 구축
x = [[0.0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
y = [[1], [1], [1], [-1], [-1], [1], [1], [1]]

# [그림 4-3(b)]의 퍼셉트론
w = tf.Variable(tf.random.uniform([3,1], -0.5, 0.5)) # 가중치: 난수 차원[2,1], 최소값 -0.5, 최대값 0.5
b = tf.Variable(tf.zeros([1]))     # 상수 1

# 옵티마이저
opt = tf.keras.optimizers.SGD(learning_rate = 0.1)

# 전방 계산(식 (4.3))
def forward():
    s = tf.add(tf.matmul(x, w), b)  # w*x + b 계산
    o = tf.tanh(s)                  # 활성함수(예측값을 확률로 변환하는 함수)로 계단함수 대신 tanh사용  
    return o                        # 계단함수는 불연속점이 있어 미분불가능, 그림 4-14 참조

# 손실 함수 정의
def loss():
    o = forward()
    return tf.reduce_mean((y-o)**2) # 손실함수 (y(실제 값) – o(예측 값))^2

# 500세대까지 학습(100세대마다 학습 정보 출력)
for i in range(2000):
    opt.minimize(loss, var_list = [w, b])
    if (i%100 == 0):
        print('loss at epoch', i, '=', loss().numpy())
        
# 학습된 퍼셉트론으로 데이터를 예측
o = forward()
print(o)

# 활성화 함수를 적용하면~~??
o = tf.sign(o)
print(o)