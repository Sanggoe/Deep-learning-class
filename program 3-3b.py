# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:42:28 2022

@author: smpsm
"""

from sklearn import datasets
import matplotlib.pyplot as plt

lfw=datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4) # 데이터셋 읽기

plt.figure(figsize=(20,5))

for i in range(8): # 처음 8명을 디스플레이
    plt.subplot(1, 8, i+1)
    plt.imshow(lfw.images[i], cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])
    
plt.show()