# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:53:39 2022

@author: smpsm
"""

데이터 증대, 과잉 적합을 방지하는 가장 확실한 방법은 큰 훈련 집합 사용
    딥러닝에서는 주어진 데이터를 인위적으로 늘리는 데이터 증대data augmentation를 적용


드롭아웃, 일정 비율의 가중치를 임의로 선택하여 불능으로 만들고 학습하는 규제 기법
    
가중치 감쇠,  성능을 유지한 채로 가중치 크기를 낮추는 규제 기법
    Kernel_regularizer(가중치에 적용) – L2 규제
    Bias_regularizer(바이어스에 적용) – 여기는..?
    Activity_regularizer(활성 함수 결과에 적용) – L1 규제

앙상블, 배치 정규화 등의 다양한 규제 기법이 있음