# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:27:53 2022

@author: smpsm
"""

import numpy as np
x2 = np.array([1,0,1])
w = np.array([-0.5,1.0,1.0])
s = np.sum(x2*w)

print(s)