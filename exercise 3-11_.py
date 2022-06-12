# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:45:53 2022

@author: smpsm
"""

from sklearn import datasets
from sklearn import svm, tree, ensemble
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits() # 필기체 1797개 샘플을 가지고 옴
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.7)
# digit.data, digit.target를 test set과 train set으로 분리하고 비율은 6:4로 지정
# 변수명은 학습데이터는 x_train, y_train 예측데이터는 x_test, y_test로 지정

s=svm.SVC(gamma=0.001)
t = tree.DecisionTreeClassifier(random_state=100)
rf = ensemble.RandomForestClassifier(random_state=100)

s.fit(x_train, y_train) # 학습 수행
t.fit(x_train, y_train) # 학습 수행
rf.fit(x_train, y_train) # 학습 수행

accu1 = cross_val_score(s, digit.data, digit.target, cv=5) # 5-겹 교차검증
accu2 = cross_val_score(t, digit.data, digit.target, cv=5) # 5-겹 교차검증
accu3 = cross_val_score(rf, digit.data, digit.target, cv=5) # 5-겹 교차검증
# cv의 값을 변화시켜 가면서 k겹 교차 검증을 수행할 수 있다.

print(accu1)
print(accu2)
print(accu3)
print("SVM 정확률(평균)=%0.3f, 표준편차=%0.3f" %(accu1.mean()*100, accu1.std()))
print("D.T 정확률(평균)=%0.3f, 표준편차=%0.3f" %(accu2.mean()*100, accu2.std()))
print("R.F 정확률(평균)=%0.3f, 표준편차=%0.3f" %(accu3.mean()*100, accu3.std()))


max_accu = max([accu1.mean(), accu2.mean(), accu3.mean()])

if max_accu == accu1.mean():
    res=s.predict(x_test)
    # 0,1,...9의 결과가 640개 생성
elif max_accu == accu2.mean():
    res=t.predict(x_test)
else:
    res=rf.predict(x_test)

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