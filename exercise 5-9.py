# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:41:54 2022

@author: smpsm
"""


연습문제 09.
[프로그램 5-7(a)]의 행 13~16은


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_train(60000, 784)
x_test(10000, 784)
y_train(60000, 10)
y_test(10000, 10)

신경망에 입력할 수 있는 형태의
x_train, x_test, y_train, y_test를 만든다.
이들 텐서의 모양을 [그림 5-5]와 같이 그리시오.