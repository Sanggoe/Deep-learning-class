# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:21:09 2022

@author: smpsm
"""

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np

''' 데이터 읽기 과정 '''
# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits() # 필기체 1797개 샘플을 가지고 옴
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)
# digit.data set을 test셋을 digit.target으로 분리하고 비율은 0.4로 지정
# 변수명은 학습데이터는 x_train, y_train 예측데이터는 x_test, y_test로 지정


''' 모델 객체 생성 '''
p = Perceptron(max_iter=1000, eta0=0.01, verbose=0)
# max_iter : Gradient Descent 방식을 반복해서 몇 번 수행할 것인가에 대한 
#            최대 반복수, 기본값 1000. 일단 수렴(Convergence)하면 횟수를
#            늘려도 성능이 거의 달라지지 않는다.
#     eta0 : 업데이터에 곱하는 값, 기본값 1 (딥러닝에 running rate와 같음)
#            너무 작으면 시간이 오래 걸리고 단순한 U자 형태의 손실함수가 아닌
#            여러 변곡점이 있는 경우에는 min값을 찾기 어려울 수 있다.
#            또 너무 크면, 수렴반대 경사로 뛰어넘는 경우가 발생
#  verbose : verbosity level(설명 수준), 훈련 과정에 대한 정보를 출력
#            기본값 0(설명 없음), 1(진행상황 출력)


''' 모델 학습 ''' 
p.fit(x_train, y_train) # digit 데이터로 모델링


''' 학습된 모델로 예측 '''
res=p.predict(x_test) # 테스트 집합으로 예측
# 0,1,...9의 결과가 640개 생성

print(len(res))

''' 성능 측정 ''' # 혼동 행렬 구함
conf=np.zeros((10,10)) 
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

# 정확률 측정 및 출력
no_correct = 0
for i in range(10):
    no_correct += conf[i][i]
accuracy = no_correct/len(res)
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")
# 퍼셉트론은 선형 분류기이기 때문에, 정확도가 SVM 모델의 98.74%보다 열등하다.