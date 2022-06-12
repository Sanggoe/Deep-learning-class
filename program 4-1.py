# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:18:33 2022

@author: smpsm
"""

from sklearn.linear_model import Perceptron

# training set 구축
X = [[0,0],[0,1],[1,0],[1,1]]
y = [-1,1,1,1] # 예제 4-1 OR data
#y = [-1,-1,-1,1] # 예제 4-1 AND data


# fit 함수로 Perceptron 학습
p = Perceptron()
p.fit(X,y)

print("학습된 퍼셉트론의 매개변수: ", p.coef_, p.intercept_) # 퍼셉트론의 가중치 w1, w2, w0
print("훈련집합에 대한 예측: ", p.predict(X))
print("정확률 측정: ", p.score(X,y)*100, "%")