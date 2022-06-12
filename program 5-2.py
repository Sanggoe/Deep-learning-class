# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:29:51 2022

@author: smpsm
"""

import tensorflow as tf
import numpy as np

t = tf.random.uniform([2,3], 0, 1)
n=np.random.uniform(0, 1, [2,3])
print("tensorflow로 생성한 텐서: \n", t, "\n")
print("numpy로 생성한 ndarray: \n", n, "\n")

res = t+n
print("덧셈 결과: \n", res)