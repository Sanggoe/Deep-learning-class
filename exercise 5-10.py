# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:11:28 2022

@author: smpsm
"""

x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
y = [[-1], [1], [1], [1]]

n_input = 2   # 입력데이터의 변수 수
n_output = 1  # 출력데이터의 변수 수

print(n_input * n_output)

x_train = (60000, 784)       # 텐서 모양 변환
x_test = (10000, 784)

n_input = 784
n_hidden = 1024
n_output = 10

print(n_input * n_hidden + n_hidden * n_output)