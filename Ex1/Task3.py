# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:08:03 2019

@author: Anton
"""

from scipy.io import loadmat
import numpy as np
import os

def main():
    os.chdir("Z:\Documents\TUT\Pattern Recognition and Machine Learning\Exercises\Ex1\Ex1_data")
    mat = loadmat("twoClassData.mat")
    X = mat['X']
    y = mat['y'].ravel()
    x = X[y==0,:]
    xs = X[y==1,:]
    plt.plot(x[:,0],x[:,1],'ro')
    plt.plot(xs[:,0],xs[:,1],'bo')
    
    
main()