# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:11:38 2022

@author: smpsm
"""

# exercise 4-13
# MLP 분류기 모델에서 은닉층 뉴런 개수를 100, 200으로 했을 때 비교
# 100으로 했을 때는 88번 수행했을 때 학습이 멈췄고 정확률은 97.82 %
# 200으로 했을 때는 63번 수행했을 때 학습이 멈췄고 정확률은 98.13 %
# 정확률은 수행할 때마다 다르겠지만, 평균적으로 은닉층 뉴런 개수가 많을 때
# 적은 반복 횟수로 학습이 완료되었다.

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np


''' 데이터 읽기 과정 '''
# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
mnist = fetch_openml('mnist_784')
mnist.data = mnist.data/255.0
x_train = mnist.data[:60000]; x_test = mnist.data[60000:]
y_train = np.int16(mnist.target[:60000]); y_test = np.int16(mnist.target[60000:])

''' 모델 객체 생성 '''
# MLP 분류기 모델을 학습      size 100 , 200 비교
mlp = MLPClassifier(hidden_layer_sizes=200, learning_rate_init=0.001,
                    batch_size=512, max_iter=300, solver='adam', verbose=True)
# hidden_layer_sizes : i번째 은닉층에서 노드 개수
#  running_rate_init : 초기 러닝레이트 설정
#         batch_size : int, default=’auto’, weight계산에 사용하는 데이터 수,
#                      여기서는 32이므로 64(8*8)바이트의 데이터를 두 개로 나누어 사용
#           max_iter : 최대 반복횟수
#             solver : 가중치 최적화 함수(τ1), sgd(stochastic gradient descent), 
#                      adam(default)
#            verbose : bool, default=False, 설명문 표시
#         activation : 활성함수(τ2), tanh, relu(default) 등
''' 모델 학습 ''' 
mlp.fit(x_train, y_train) # 학습

''' 학습된 모델로 예측 '''
res=mlp.predict(x_test) # 테스트 집합으로 예측

''' 성능 측정 '''
# 혼동 행렬 구함
conf=np.zeros((10,10)) 
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

# 정확률 계산
no_correct = 0
for i in range(10):
    no_correct += conf[i][i]
accuracy = no_correct/len(res)
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")
# 정확도는 SVM 모델의 98.74%보다 열등하지만, 이 방법이 빠르다고 한다.