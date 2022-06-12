# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:08:52 2022

@author: smpsm
"""

from sklearn import datasets
import matplotlib.pyplot as plt

digit=datasets.load_digits()

plt.figure(figsize=(5,5))   # 그림의 크기 설정
for i in range(10):
    plt.imshow(digit.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    # 그림 보여주기 설정: 첫번째 이미지를 cmap=스칼라데이터를 색상에 매핑, 보간법은 가장 가까운 값
    plt.show()
    print(digit.data[i])    # 0번 샘플의 화솟값을 출력
    print("이 숫자는 ", digit.target[i],"입니다.") # 기계학습으로 예측하고자 하는 값