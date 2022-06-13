# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:18:54 2022

@author: smpsm
"""

연습문제 11. [그림 4-12(b)]에 있는 퍼셉트론 ③의 가중치가 (-1.0, 1.0, 1.0)에서
(-0.9, 0.9, 0.8)로 바뀌었다고 가정하자. 새로운 다층 퍼셉트론에 대해 [예제 4-3]을 수행하시오

-0.9  0.9  0.8 로 바꾸어 계산 과정 따라가면

-1  0.8  0.8  -0.8



연습문제 12. [그림 4-12(b)]의 다층 퍼셉트론에서 u1_21, u1_12, u2_12는 얼마인지 적으시오.

입력층(i)  →  은닉층(j)  →  출력층(k)
	u1(ji)	     u2(kj)

u1_21 : z2에 들어오는 입력 x1의 가중치 = -1.0
u1_12 : z1에 들어오는 입력 x2의 가중치 = 1.0
u2_12 : 출력에 들어오는 입력 z2의 가중치 = 1.0
