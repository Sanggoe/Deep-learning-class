# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:41:01 2022

@author: smpsm
"""

import time

start = time.time()
sum=0
for i in range(1,100000001):
    sum += i
    
end = time.time()

print('1+2+...+100000000=', sum)
print('소요 시간은 ', end-start, '초입니다.')