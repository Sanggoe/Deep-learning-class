# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:23:20 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# 앞서 선행되었어야 할 코드
# cnn.save("my_cnn.h5")

# 신경망 구조와 가중치를 저장하고 있는 파일을 읽어옴
cnn = tf.keras.models.load_model("my_cnn.h5")
cnn.summary()

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

res = cnn.evaluate(x_test, y_test, verbose=0)
print("정확률은", res[1]*100)
