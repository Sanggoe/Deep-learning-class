# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:01:39 2022

@author: smpsm
"""

'''
연습문제 02. 그림 4-3에서 샘플 4개에 대해 제대로 인식하는지 확인하시오.
가중치 w0=-0.5, w1=1, w2=1

x1 = (0,0) 샘플을 예측해보면, o= 𝛕(w1x1+w2x2+w0) = 𝛕(1*0+1*0-0.5) = 𝛕(-0.5) = -1
x2 = (0,1) 샘플을 예측해보면, o= 𝛕(w1x1+w2x2+w0) = 𝛕(1*0+1*1-0.5) = 𝛕(0.5) = 1
x3 = (1,0) 샘플을 예측해보면, o= 𝛕(w1x1+w2x2+w0) = 𝛕(1*1+1*0-0.5) = 𝛕(0.5) = 1
x4 = (1,1) 샘플을 예측해보면, o= 𝛕(w1x1+w2x2+w0) = 𝛕(1*1+1*1-0.5) = 𝛕(1.5) = 1

y1부터 y4까지의 레이블 역시 활성함수의 결과값과 같으므로, 제대로 인식한다고 볼 수 있다.
'''