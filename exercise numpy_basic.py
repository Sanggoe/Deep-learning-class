# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 23:26:53 2022

@author: smpsm
"""

''' 2.2 Basics '''

[[1., 0., 0.],
 [0., 1., 2.]]

import numpy as np
a = np.arange(15).reshape(3,5)
print(a)
print()

print('a.shape\n', a.shape)
print()

print('a.ndim\n', a.ndim)
print()

print('a.dtype.name\n', a.dtype.name)
print()


print('a.itemsize\n', a.itemsize)
print()

print('a.size\n', a.size)
print()

print('type(a)\n', type(a))
print()

b = np.array([6, 7, 8])
print(b)
print(type(b))


''' 2.2.2 Array Creation '''
import numpy as np
a = np.array([2,3,4])
a

a.dtype

b = np.array([1.2, 3.5, 5.1])
b.dtype

# Wrong
# a = np.array(1, 2, 3, 4)

# Right
a = np.array([1, 2, 3, 4])

b = np.array([(1.5, 2, 3), (4, 5, 6)])
b

b = np.array([(1.5, 2, 3), (4, 5, 6)])
b

# dtype은 초기화 시 지정 가능
c = np.array([[1, 2], [3, 4]], dtype=complex)
c

# 0으로 초기화
np.zeros((3, 4))

# 1로 초기화
np.ones((2, 3, 4), dtype=np.int16)

# 빈 array
np.empty((2, 3))

np.arange(10, 30, 5) # 10부터 30까지 5씩 증가

np.arange(0, 2, 0.3) # it accepts float arguments

from numpy import pi
np.linspace(0, 2, 9)  # 0부터 2까지 9개로 나누기

x = np.linspace(0, 2 * pi, 100)

f = np.sin(x)


''' 2.2.3 Printing Arrays '''
a = np.arange(6) # 1d array
print(a)

b = np.arange(12).reshape(4, 3) # 2d array
print(b)

c = np.arange(24).reshape(2, 3, 4) # 3d array
print(c)


''' 3.13 Indexing and slicing '''
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(a[a < 5])

five_up = (a >= 5)
five_up

print(a[five_up])

divisible_by_2 = a[a%2==0]
print(divisible_by_2)


''' 3.15 Basic array operations '''
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
data + ones

data * data

data * ones

data / data

data // data

a = np.array([1, 2, 3, 4])
a.sum()

b = np.array([[10, 10], [0, 1]])
b.sum(axis=0)

b.sum(axis=1)


''' 3.16 Broadcasting '''
data = np.array([1.0, 2.0])
data * 1.6


''' 3.17 More useful array operations '''
data = np.array([1, 2, 3])

data.max()

data.min()

data.sum()

a = np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
            [0.54627315, 0.05093587, 0.40067661, 0.55645993],
            [0.12697628, 0.82485143, 0.26590556, 0.56917101]])
    
a.sum()

a.min()

a.min(axis=0) # axis = 0 은, 세로줄 비교

a.min(axis=1) # axis = 1 은, 가로줄 비교


''' 3.18 Creating matrices '''
data = np.array([[1, 2], [3, 4], [5, 6]])
data

data[0, 1]

data[1:3]

data[0:2, 0]

data.max()

data.min()

data.sum()

data = np.array([[1, 2], [5, 3], [4, 6]])
data

data.max(axis=0) # axis = 0 은, 세로줄 비교

data.max(axis=1) # axis = 1 은, 가로줄 비교

data = np.array([[1, 2], [3, 4]])
ones = np.array([[1, 1], [1, 1]])
data + ones

data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
ones_row

data + ones_row

np.ones((4, 3, 2))

np.ones(3)

np.zeros(3)

rng = np.random.default_rng(0)
rng.random(3)