from sklearn.linear_model import LogisticRegression
import os
import numpy as np

os.chdir('Z:\Documents\TUT\Pattern Recognition and Machine Learning\Exercises\Ex4')    
# 1) Load X and y.   
X = np.array(np.genfromtxt('X.csv',delimiter=','))    
y = np.array(np.genfromtxt('y.csv'))

# 2) Initialize w at w = np.array([1, -1])
w = np.array([1,-1])

# 3) Set step_size to a small positive value.
step_size = 0.01

clf = LogisticRegression()
clf.fit(X[:,np.newaxis],y)
