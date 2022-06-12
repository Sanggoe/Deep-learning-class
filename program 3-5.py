# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:43:43 2022

@author: smpsm
"""

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits() # 필기체 1797개 샘플을 가지고 옴
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)
# digit.data, digit.target를 test set과 train set으로 분리하고 비율은 6:4로 지정
# 변수명은 학습데이터는 x_train, y_train 예측데이터는 x_test, y_test로 지정

s=svm.SVC(gamma=0.001)
s.fit(x_train, y_train) # 학습 수행

res=s.predict(x_test)
# 0,1,...9의 결과가 640개 생성

# 혼동 행렬 구함
conf=np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 # res와 y_test가 같으면 +1 하라는 의미
print(conf)

# 정확률 측정하고 출력
no_correct = 0
for i in range(10):
    no_correct += conf[i][i] # 혼동행렬 대각선이 참값이니까..
accuracy = no_correct/len(res)
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")