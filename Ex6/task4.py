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

clf = LogisticRegression()


mat = loadmat('arcene\\arcene.mat')
#print(mat.keys())
X_test = mat['X_test']
X_train = mat['X_train']
y_test = mat['y_test'].ravel()
y_train = mat['y_train'].ravel()

rfe = RFECV(estimator=clf, step=50, verbose=1)

rfe.fit(X_train, y_train)

selected = rfe.support_

print('Optimal # features:', np.sum(selected)) # 8600
plt.plot(range(0,10001,50),rfe.grid_scores_)
plt.show()
preds=rfe.predict(X_test)
print('Accuracy is:',accuracy_score(preds,y_test)) #accuracy 0.84