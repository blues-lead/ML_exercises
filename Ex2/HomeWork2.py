# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:21:11 2019

@author: Anton
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    os.chdir("Z:\Documents\TUT\Pattern Recognition and Machine Learning\Exercises\Ex2\Ex1_data")
    mat = loadmat("twoClassData.mat")
    X = mat['X']
    y = mat['y'].ravel()
    train, test, train_y, test_y = train_test_split(X,y,test_size=0.5)
    model = KNeighborsClassifier()
    model.fit(train,train_y)
    pred_y = model.predict(test)
    print(accuracy_score(test_y,pred_y))
    
    model = LinearDiscriminantAnalysis()
    model.fit(train,train_y)
    model.fit(train,train_y)
    pred_y = model.predict(test)
    print(accuracy_score(test_y,pred_y))
    
    
    
main()