# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:27:24 2022

@author: smpsm
"""

from sklearn.linear_model import Perceptron

''' 데이터 읽기 과정 '''
# training set 구축
X = [[0,0],[0,1],[1,0],[1,1]] # 예제 4-1 Or data
y = [-1,1,1,1]


''' 모델 객체 생성 '''
# fit 함수로 Perceptron 학습
p = Perceptron()


''' 모델 학습 ''' 
p.fit(X,y)


''' 학습된 모델로 예측 '''
print("학습된 퍼셉트론의 매개변수: ", p.coef_, p.intercept_) # 퍼셉트론의 가중치 w1, w2, w0
print("훈련집합에 대한 예측: ", p.predict(X))


''' 성능 측정 '''
print("정확률 측정: ", p.score(X,y)*100, "%")