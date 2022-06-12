# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:42:58 2022

@author: smpsm
"""

from sklearn import datasets

news=datasets.fetch_20newsgroups(subset='train') # 데이터셋 읽기
print("*****\n", news.data[0], "\n*****")   # 0번 샘플 출력
print("이 문서의 부류는 <", news.target_names[news.target[0]], "> 입니다.")

# print(news.values())