# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:29:21 2019

@author: Anton
"""

from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore') 

clf = LogisticRegression(penalty='l1')

C_array = 10 ** np.arange(0,15,0.5)

mat = loadmat('arcene\\arcene.mat')
#print(mat.keys())
X_test = mat['X_test']
X_train = mat['X_train']
y_test = mat['y_test'].ravel()
y_train = mat['y_train'].ravel()

opt_c = 0
score = 0
for c in C_array:
    clf.C = c
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    #print('# of selected features:', clf.coef_)
    if np.mean(cross_val_score(clf, X_test, y_test)) > score:
        score = np.mean(cross_val_score(clf, X_test, y_test))
        opt_c = c

print('Max score:',score,'C-value:', opt_c)

clf.C = opt_c
clf.fit(X_train, y_train)
f_selected = np.count_nonzero(clf.coef_)
print('Features selected:',f_selected)
print('Accuracy score:', np.mean(cross_val_score(clf,X_test,y_test)))