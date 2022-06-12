# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:07:58 2022

@author: smpsm
"""

import tensorflow as tf

print(tf.__version__)
a = tf.random.uniform([2,3], 0, 1)
print(a)
print(type(a))