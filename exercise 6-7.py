# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:42:23 2022

@author: smpsm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)/255.0  # ndarray로 변환
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) # 원핫 코드로 변환
y_test = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()

# (1) 세대수 충분히 늘려 학습
'''
C-C-P-C-C-P 구조인 경우이고, epoch을 100으로 두어 학습을 진행하였다.
그랬을 때, 에폭이 24정도 되었을 때 처음으로 가장 낮은 오차를 기록했고,
그 이후로 계속해서 진동하며 수렴하는 것을 알 수 있었다.
당시 테스트 데이터의 정확률은 79%에 해당한다.

Epoch 24/100
391/391 - 161s - loss: 0.4891 - accuracy: 0.8250 - val_loss: 0.6214 - val_accuracy: 0.7907 - 161s/epoch - 411ms/step

정확률은 80.47000169754028
'''




# (2) 구조 변경해가며 성능 비교
'''
출력된 결과를 놓고 비교해보았을 때는, C-C-P-C-C-P 구조가
가장 높은 정확률을 보였기 때문에 제일 효율적인 디자인 패턴으로 볼 수 있었다.
'''


'''
# C C P  C C P
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(Conv2D(32,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
'''
'''
# C C P
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(Conv2D(32,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
'''
'''
# C C P  C C P  C C P
cnn.add(Conv2D(16,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(Conv2D(16,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(Conv2D(32,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
'''
'''
# C P C P
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
'''

# C P C P C P
cnn.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) 
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))


cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))


# 신경망 학습
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
hist=cnn.fit(x_train, y_train, batch_size=128, epochs=12,
             validation_data=(x_test, y_test), verbose=2)

res=cnn.evaluate(x_test, y_test, verbose=0)
print("정확률은", res[1]*100)

import matplotlib.pyplot as plt

# 정확률 곡선
plt.plot(hist.history['accuracy'])     # train set에 대한 정확성 그래프
plt.plot(hist.history['val_accuracy']) # test set에 대한 정확성 그래프
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left') # 범례, loc(위치): 왼쪽 위
plt.grid()                                            # grid 그리기
plt.show()                                            # 그래프 그리기

# 손실 함수 곡선
plt.plot(hist.history['loss'])     # train set에 대한 오차 그래프
plt.plot(hist.history['val_loss']) # test set에 대한 오차 그래프
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show() 

cnn.save("my_cnn.h5")