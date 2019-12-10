# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:48:51 2019

@author: hehu
"""

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern

def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

# Test our loader

#X, y = load_data(".")
X, y = load_data("Z:\Documents\TUT\Pattern Recognition and Machine Learning\Exercises\Ex3\GTSRB_subset\GTSRB_subset")
np.save('gtsrb_X',X)
np.save('gtsrb_y',y)
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))

# Continue your code here...
#============================= Task 4/5 =========================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def warn(*args, **kwargs): #suppress warnings from sklearn
    pass #suppress warnings from sklearn
import warnings #suppress warnings from sklearn
warnings.warn = warn #suppress warnings from sklearn

clfs = {}
clfs['kneighbor'] = {'alg':KNeighborsClassifier() ,'score':0, 'stdev':0}
clfs['svm'] = {'alg':SVC(), 'score':0, 'stdev':0}
clfs['logreg'] = {'alg':LogisticRegression(), 'score':0, 'stdev':0}
clfs['lda'] = {'alg':LinearDiscriminantAnalysis(),'score':0, 'stdev':0}
clfs['random_forest'] = {'alg':RandomForestClassifier(),'score':0,'stdev':0}
clfs['extra_forest'] = {'alg':ExtraTreesClassifier(),'score':0,'stdev':0}
clfs['ada_boost'] = {'alg':AdaBoostClassifier() ,'score':0,'stdev':0}
clfs['gradient_boost'] = {'alg':GradientBoostingClassifier(),'score':0, 'stdev':0}

for key in clfs:
    clfs[key]['alg'].fit(F,y.ravel())
    scores_vector = cross_val_score(clfs[key]['alg'],F,y)
    clfs[key]['score'] = scores_vector.mean()
    clfs[key]['stdev'] = scores_vector.std()*2
print()
print("{:^20s}{:^20s}{:^20}".format("Algorithm","Scores","+/- deviation"))    
print("="*60)
for key in clfs:
    print("{:^20s}{:^20.2f}{:^20.2f}".format(key,clfs[key]['score'],clfs[key]['stdev']))
print("="*60)
#==============================================================================
    