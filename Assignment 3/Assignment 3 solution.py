# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sc
import os, csv, cv2, math, itertools, random
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.linear_model import LinearRegression

wd = 'C:/Users/sheng/Downloads/class (teaching)/EGR 598 ML & AI/Assignment' # directory to save results

data = pd.read_csv(wd+'/transfusion.data', sep=",")
data = pd.DataFrame(data).to_numpy()

Cls = data[:, 4].astype('int')
X = data[:, 0:4].astype('float')

''' Part 1 '''
### P1-1
k = 2
thresh = 0.001

diff = np.ones(k)
centers = np.random.normal(size=(k, X.shape[1])) # initialize with normal random numbers
centers0 = np.random.normal(size=(k, X.shape[1]))

count = 0
while len(np.where(diff>=thresh)[0])>0:    
    cl = []
    for x in X:
        temp = x-centers
        D = np.mean(np.matmul(temp, np.transpose(temp)), 1)
        ind = np.argmin(D)
        cl.append(ind)
        
        ind1 = np.where(cl==ind)[0]
        if len(ind1)>0:
            centers[ind, :] = np.mean(X[ind1, :], 0)
        else:
            centers[ind, :] = x
    
    diff = np.mean(np.matmul(centers-centers0, np.transpose(centers-centers0)), 1)
    centers0 = centers
    count += 1

### P1-3
M1 = confusion_matrix(Cls, cl, labels=np.arange(2).astype('int')) 
print(M1) 
print(np.trace(M1)/np.sum(M1))

''' Part 2 '''
### P2-1
perm = np.array(random.sample(list(np.arange(X.shape[0])), X.shape[0]))
X1, Cls1 = X[perm, :], Cls[perm]

xtrain, xtest = X1[0:400, :], X1[400:X.shape[0], :]
ytrain, ytest = Cls1[0:400], Cls1[400:X.shape[0]]

NB = GaussianNB().fit(xtrain, ytrain)
pred = NB.predict(xtest)

M1 = confusion_matrix(ytest, pred, labels=np.arange(2).astype('int')) 
print(M1) 
print(np.trace(M1)/np.sum(M1))

### P2-3
model = svm.SVC(kernel='rbf')
model.fit(xtrain, ytrain)

pred = model.predict(xtest)

M2 = confusion_matrix(ytest, pred, labels=np.arange(2).astype('int')) 
print(M2) 
print(np.trace(M2)/np.sum(M2))

''' Part 3 '''
### P3-1
clf = RandomForestClassifier(n_estimators=500, random_state=0).fit(xtrain, ytrain)
pred = clf.predict(xtest)

M1 = confusion_matrix(ytest, pred, labels=np.arange(2).astype('int')) 
print(M1) 
print(np.trace(M1)/np.sum(M1))

### P3-2
clf = AdaBoostClassifier(n_estimators=500, random_state=0).fit(xtrain, ytrain)
pred = clf.predict(xtest)

M1 = confusion_matrix(ytest, pred, labels=np.arange(2).astype('int')) 
print(M1) 
print(np.trace(M1)/np.sum(M1))

''' Part 4 '''
### P4-1
xtrain, xtest = X1[0:400, [0, 2, 3]], X1[400:X.shape[0], [0, 2, 3]]
ytrain, ytest = X1[0:400, 1], X1[400:X.shape[0], 1]

m = np.mean(xtrain, 0)
S = np.cov(np.transpose(xtrain))
print(m)
print(S)

### P4-2
kernel = kernels.DotProduct() + kernels.WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(xtrain, ytrain)
print(gpr.score(xtrain, ytrain))
pred = gpr.predict(xtest)

MSE = np.mean((pred-ytest)**2)
print(MSE)

### P4-3
reg = LinearRegression().fit(xtrain, ytrain)
print(reg.score(xtrain, ytrain))
pred = reg.predict(xtest)

MSE = np.mean((pred-ytest)**2)
print(MSE)