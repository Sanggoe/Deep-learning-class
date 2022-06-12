# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:41:24 2022

@author: smpsm
"""

import tensorflow as tf

# OR 데이터 구축
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
y = [[-1], [1], [1], [1]]

# [그림 4-3(b)]의 퍼셉트론
w = tf.Variable([[1.0], [1.0]]) # weight Matrix
b = tf.Variable(-0.5)           # 상수 벡터

# 식 4.3의 퍼셉트론 동작
s = tf.add(tf.matmul(x, w), b)  # w*x + b, matmul(matrix multiplication)
o = tf.sign(s)                  # 활성함수 sign(), 음수는 -1, 양수는 1

print(o)