# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:41:54 2022

@author: smpsm
"""


연습문제 09.
[프로그램 5-7(a)]의 행 13~16은


x_train = x_train.reshape(60000, 784)       # 텐서 모양 변환
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0


신경망에 입력할 수 있는 형태의
x_train, x_test, y_train, y_test를 만든다.
이들 텐서의 모양을 [그림 5-5]와 같이 그리시오.