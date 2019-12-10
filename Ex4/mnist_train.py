# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:47:06 2019

@author: Anton
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, MaxPooling2D, Conv2D
from keras.utils import to_categorical
import numpy as np
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

# Keras assumes 4D input -> add a dummy axis !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
X_train = X_train[...,np.newaxis] /255.0
X_test = X_test[...,np.newaxis] /255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_featmaps = 32
num_classes = 10
num_epochs = 20
w,h=5,5

model = Sequential()

#Layer 1
model.add(Conv2D(num_featmaps,(w,h),input_shape=(28,28,1), activation='relu'))

#Layer 2
model.add(Conv2D(num_featmaps,(w,h),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 3
#Flatten() vectorizes the data: 32x10x10 -> 3200
#(10x10 instead of 14x14 due to border effect)
model.add(Flatten())
model.add(Dense(128,activation='relu'))

#Layer 4
#Last layer producing outputs
model.add(Dense(num_classes,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test,y_test))

#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

model.save('mnst.h5')

## load model
#model = load_model('model.h5')
## summarize model.
#model.summary()
## load dataset
#dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## evaluate the model
#score = model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))