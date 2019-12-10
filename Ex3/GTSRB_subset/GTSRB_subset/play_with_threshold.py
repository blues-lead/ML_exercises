# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.stats as stats


# %%
mean1=np.array((1,1)).T
cov1 = np.array(([1,0],[0,1]))
x1,y1 = np.random.multivariate_normal(mean1,cov1,100).T
plt.plot(x1,y1,'bo')

mean2=np.array((3,3)).T
cov2 = np.array(([2,0],[0,2]))
x2,y2 = np.random.multivariate_normal(mean2,cov2,100).T
plt.plot(x2,y2,'ro')
#====================
temp1 = np.append(x1,x2)
temp2 = np.append(y1,y2)
X = np.vstack((temp1,temp2)).T
y = np.vstack((np.zeros((100,1)),np.ones((100,1))))
clf = LinearDiscriminantAnalysis()
clf.fit(X,y.ravel())
#print(clf.predict
x = range(-2,6)
yt = clf.coef_[0][0]*x + clf.coef_[0][1]
plt.plot(x,yt,color='magenta')
print(clf.intercept_)
mCov = np.linalg.inv(cov1+cov2)
print("mu",mean1.shape, "covariance", mCov.shape, "mu",mean1.T.shape)
#c = -0.5*np.dot(mean2.T,mCov,mean2)+(0.5)*np.dot(mean1.T,mCov,mean2)
#print("c =", c)
#rint(mCov)

