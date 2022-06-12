# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:11:41 2022

@author: smpsm
"""

''' 프로그램 3-1(a) '''
from sklearn import datasets

d=datasets.load_iris()  # iris 데이터셋 읽고
print(d.DESCR)          # 내용 출력


''' 프로그램 3-1(b) '''
for i in range(0,len(d.data)):      # 샘플을 순서대로 출력
    print(i+1, d.data[i], d.target[i])
    
    
''' 프로그램 3-1(c) '''
from sklearn import svm

s=svm.SVC(gamma=0.1, C=10) # svm 분류 모델 SVC 객체 생성
# gamma: ‘rbf’(Radial Basis Function), ‘poly’
#  and ‘sigmoid’에 대한 커널변수로 숫자 또는 auto  
# C: 정규화 변수, 정규화의 강도는 C에 반비례 
s.fit(d.data, d.target) # iris 데이터로 학습 # train set

new_d=[[6.4,3.2,6.0,2.5],[7.1,3.1,4.7,1.35]]
# 각각 2의 값 / 1의 값에 가깝도록 새로운 데이터

res=s.predict(new_d) # Test set // 예측할 때 사용할 데이터
print("새로운 2개 샘플의 부류는", res)

# 원핫 코드는 한 요소만 1인 이진열을 말함
# train set으로 모델링과 test set으로 예측을 수행