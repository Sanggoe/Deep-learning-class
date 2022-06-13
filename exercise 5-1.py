# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:09:39 2022

@author: smpsm
"""

# exercise 5-1

import tensorflow as tf
import tensorflow.keras.datasets as ds

# Boston Housing 읽고 텐서 모양 출력
(x_train, y_train), (x_test, y_test) = ds.boston_housing.load_data()
print("Boston Housing shape: ", x_train.shape, y_train.shape)
print("contents: \n", x_train, y_train)

# Reuters읽고 텐서 모양 출력
(x_train, y_train), (x_test, y_test) = ds.reuters.load_data()
print("Reuters shape: ", x_train.shape, y_train.shape)
print("contents: \n", x_train, y_train)
