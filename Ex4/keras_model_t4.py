# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:45:52 2019

@author: Anton
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os



N = 32
w,h = 5,5
X = np.load('gtsrb_X.npy')
X = X[..., np.newaxis] / 255.0
y = np.load('gtsrb_y.npy')
y = np_utils.to_categorical(y)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv2D(N,(w,h),input_shape=(64,64,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Conv2D(N, (w,h), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(2,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20, batch_size=32 ,validation_data=(X_test,y_test))