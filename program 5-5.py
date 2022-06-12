# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:02:03 2022

@author: smpsm
"""

import tensorflow as tf

# OR 데이터 구축
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]] # 입력변수
y = [[-1], [1], [1], [1]]                            # 출력변수

# [그림 4-3(b)]의 퍼셉트론
w = tf.Variable(tf.random.uniform([2,1], -0.5, 0.5)) # 가중치: 난수 차원[2,1], 최소값 -0.5, 최대값 0.5
b = tf.Variable(tf.zeros([1]))     # 상수 1

# 옵티마이저
opt = tf.keras.optimizers.SGD(learning_rate = 0.1)

# 전방 계싼(식 (4.3))
def forward():
    s = tf.add(tf.matmul(x, w), b)  # w*x + b 계산
    o = tf.tanh(s)                  # 활성함수(예측값을 확률로 변환하는 함수)로 계단함수 대신 tanh사용  
    return o                        # 계단함수는 불연속점이 있어 미분불가능, 그림 4-14 참조

# 손실 함수 정의
def loss():
    o = forward()
    return tf.reduce_mean((y-o)**2) # 손실함수 (y(실제 값) – o(예측 값))^2

# 500세대까지 학습(100세대마다 학습 정보 출력)
for i in range(500):
    opt.minimize(loss, var_list = [w, b])
    if (i%100 == 0):
        print('loss at epoch', i, '=', loss().numpy())
        
# 학습된 퍼셉트론으로 OR 데이터를 예측
o = forward()
print(o)

# 활성화 함수를 적용하면~~??
o = tf.sign(o)
print(o)