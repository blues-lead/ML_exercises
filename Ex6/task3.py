# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:09:33 2019

@author: Anton
"""


from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

clf = RandomForestClassifier(n_estimators=100)


mat = loadmat('arcene\\arcene.mat')
#print(mat.keys())
X_test = mat['X_test']
X_train = mat['X_train']
y_test = mat['y_test'].ravel()
y_train = mat['y_train'].ravel()

clf.fit(X_train, y_train)

importancies=clf.feature_importances_

print(X_train.shape)

#for f in range(X_train.shape[1]):
#    if importancies[indecies[f]] > 0.001:
#        print("%d. feature %d (%f)" % (f + 1, indecies[f], importancies[indecies[f]]))

plt.figure()
plt.title("Feature importances")
#plt.bar(range(X_train.shape[1]), importancies[indecies],
#       color="r")
plt.bar(np.arange(len(importancies)), importancies, color="r")
plt.show()

