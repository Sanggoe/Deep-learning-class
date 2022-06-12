# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:20:37 2022

@author: smpsm
"""

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

digit=datasets.load_digits()
s=svm.SVC(gamma=0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv=10) # 10-겹 교차검증
# cv의 값을 변화시켜 가면서 k겹 교차 검증을 수행할 수 있다.

print(accuracies)
print("정확률(평균)=%0.3f, 표준편차=%0.3f" %(accuracies.mean()*100, accuracies.std()))