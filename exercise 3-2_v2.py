# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:41:08 2022

@author: smpsm
"""
    
from sklearn import svm
from sklearn import datasets
from random import randint, random, choice

d=datasets.load_iris()  # iris 데이터셋 읽고

s=svm.SVC(gamma=0.1, C=10) # svm 분류 모델 SVC 객체 생성
s.fit(d.data, d.target) # iris 데이터로 학습 # train set

n_data = len(d.data)
test_set=[]
test_target=[]

for i in range(20):    
    rand_index = randint(0, n_data-1) # random index 반환
    
    new_data = d.data[rand_index]
    new_target = d.target[rand_index]
    
    new_data += new_data*((random()-0.5)*0.1) # 5% 이내에서 값 랜덤 수정
    
    test_set.append(new_data)
    test_target.append(new_target)
    
    
res=s.predict(test_set) # Test set // 예측할 때 사용할 데이터
accuracy_count = 0

print("새로운 20개 샘플의 부류는")
print("\t [test data] / test target / result")
for i in range(len(test_set)):  # 샘플을 순서대로 출력
    print("%2d" % (i+1), test_set[i], "/", test_target[i], "/", res[i])
    
    # 정확률 측정
    if test_target[i] == res[i]:
        accuracy_count += 1

print("정확률은 %lf" % (accuracy_count/len(res)*100))
    

# 원핫 코드는 한 요소만 1인 이진열을 말함
# train set으로 모델링과 test set으로 예측을 수행