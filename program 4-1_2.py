# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:35:41 2022

@author: smpsm
"""

import numpy as np
x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
w = np.array([-0.5,1.0,1.0])
s = np.sum(x*w, axis=1)

print(s)