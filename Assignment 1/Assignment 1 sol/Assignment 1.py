# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, csv, cv2, math, itertools, random
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from matplotlib.ticker import PercentFormatter

wd = 'C:/Users/sheng/Downloads/class (teaching)/EGR 598 ML & AI/Assignment' # directory to save results

data = pd.read_csv(wd+'/Raisin_Dataset/Raisin_Dataset.csv')
data = pd.DataFrame(data).to_numpy()

Cls = data[:, data.shape[1]-1] # class
X = data[:, 0:(data.shape[1]-1)].astype('float') # matrix of all input attributes

''' Part 1 '''
### P1-3
data1 = data[np.where(Cls=='Besni')[0], :]
data2 = data[np.where(Cls=='Kecimen')[0], :]
data1 = data1[:, 0:(data1.shape[1]-1)].astype('float')
data2 = data2[:, 0:(data2.shape[1]-1)].astype('float')

m1 = np.mean(data1, 0)
m2 = np.mean(data2, 0)
Cov1 = np.cov(np.transpose(data1))
Cov2 = np.cov(np.transpose(data2))

print(m1), print('\n')
print(m2),  print('\n')
print(Cov1), print('\n')
print(Cov2),  print('\n')

### P1-4
sample1 = np.random.multivariate_normal(m1, Cov1, 10)
sample2 = np.random.multivariate_normal(m2, Cov2, 10)

print(sample1), print('\n')
print(sample2), print('\n')

### P1-5
m1a = m1[[1, 2]]
m2a = m2[[1, 2]]

Cov1a = Cov1[[1, 2], :]
Cov2a = Cov2[[1, 2], :]
Cov1a = Cov1a[:, [1, 2]]
Cov2a = Cov2a[:, [1, 2]]

x = np.linspace(0,800,500) 
y = np.linspace(0,800,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

rv = multivariate_normal(m1a, Cov1a) # class 1

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

rv = multivariate_normal(m2a, Cov2a) # class 2

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

### P1-7
def labeling(x, m1, Cov1, m2, Cov2, P): # P is the estimated probability of class 1
    p1 = multivariate_normal.pdf(x, mean=m1, cov=Cov1, allow_singular=True)
    p2 = multivariate_normal.pdf(x, mean=m2, cov=Cov2, allow_singular=True)
    g1 = np.log(p1)+np.log(P)
    g2 = np.log(p2)+np.log(1-P)
    
    label = 'Besni' if g1>g2 else 'Kecimen'
    
    return label

labels = []
for i in X:
    lb = labeling(i, m1, Cov1, m2, Cov2, 0.5)
    labels.append(lb)

d = {'labels': labels}
df = pd.DataFrame(d)
df.to_csv(wd+'/A1-P1(7) labels.csv', index=False)

### P1-8
def labeling_pool(x, m1, m2, S, P): # P is the estimated probability of class 1
    g1 = -0.5*np.matmul(np.matmul(x-m1, np.linalg.inv(S)), x-m1)+np.log(P)
    g2 = -0.5*np.matmul(np.matmul(x-m2, np.linalg.inv(S)), x-m2)+np.log(1-P)
    
    label = 'Besni' if g1>g2 else 'Kecimen'
    
    return label

S = 0.5*(Cov1+Cov2)

labels_pool = []
for i in X:
    lb = labeling_pool(i, m1, m2, S, 0.5)
    labels_pool.append(lb)

d = {'labels': labels_pool}
df = pd.DataFrame(d)
df.to_csv(wd+'/A1-P1(8) labels.csv', index=False)

### P1-9
M1 = confusion_matrix(Cls, labels, labels=['Besni', 'Kecimen']) # Cls from P1-3 is the ground truth
M2 = confusion_matrix(Cls, labels_pool, labels=['Besni', 'Kecimen'])
print(M1) 
print(M2) 

acc1 = np.trace(M1)/np.sum(M1)
acc2 = np.trace(M2)/np.sum(M2)

''' Part 2 '''
K = 4

perm = np.array(random.sample(list(np.arange(X.shape[0])), X.shape[0]))
run = int(X.shape[0]/K)

TP1, TN1, FP1, FN1 = [], [], [], []
TP2, TN2, FP2, FN2 = [], [], [], []
for i in range(K):
    ind_te = perm[i*run:(i+1)*run] # index of testing data
    ind_tr = np.delete(perm, ind_te) # index of training data

    xtrain, xtest = X[ind_tr, :], X[ind_te, :]
    ytrain, ytest = Cls[ind_tr], Cls[ind_te]

    index1 = np.where(ytrain=='Besni')[0]
    index2 = np.where(ytrain=='Kecimen')[0]
    
    m1 = np.mean(xtrain[index1, :], 0)
    m2 = np.mean(xtrain[index2, :], 0)
    Cov1 = np.cov(np.transpose(xtrain[index1, :]))
    Cov2 = np.cov(np.transpose(xtrain[index2, :]))
    
    S = (len(index1)*Cov1+len(index2)*Cov2)/(len(index1)+len(index2))
    
    labels, labels_pool = [], []
    for j in xtest:
        lb1 = labeling(j, m1, Cov1, m2, Cov2, len(index1)/(len(index1)+len(index2))) # use the functions you defined for part 1
        lb2 = labeling_pool(j, m1, m2, S, len(index1)/(len(index1)+len(index2)))
        labels.append(lb1)
        labels_pool.append(lb2)
    
    d1 = {'labels': labels}
    df = pd.DataFrame(d1)
    df.to_csv(wd+'/A1-P2(2) labels_CV'+str(i+1)+'.csv', index=False)
    
    d2 = {'labels': labels_pool}
    df = pd.DataFrame(d2)
    df.to_csv(wd+'/A1-P2(3) labels_CV'+str(i+1)+'.csv', index=False)
    
    index1a = np.where(ytest=='Besni')[0]
    index2a = np.where(ytest=='Kecimen')[0]

    M1 = confusion_matrix(ytest, labels, labels=['Besni', 'Kecimen']) # confusion matrix for 2
    M2 = confusion_matrix(ytest, labels_pool, labels=['Besni', 'Kecimen']) # confusion matrix for 3
    print(M1) 
    print(M2) 
    
    tn1, fp1, fn1, tp1 = M1.ravel()
    tn2, fp2, fn2, tp2 = M2.ravel()
    
    TP1.append(tp1/(tp1+fn1))
    TN1.append(tn1/(tn1+fp1))
    FP1.append(fp1/(tn1+fp1))
    FN1.append(fn1/(tp1+fn1))
    TP2.append(tp2/(tp2+fn2))
    TN2.append(tn2/(tn2+fp2))
    FP2.append(fp2/(tn2+fp2))
    FN2.append(fn2/(tp2+fn2))

print(np.mean(TP1))
print(np.mean(TN1))
print(np.mean(FP1))
print(np.mean(FN1))
print(np.mean(TP2))
print(np.mean(TN2))
print(np.mean(FP2))
print(np.mean(FN2))

''' Part 3 '''
### P3-1
X1 = data[:, [0, 1, 2, 3, 5, 6]].astype('float')
Y = data[:, 4].astype('float')

for i in range(X1.shape[1]):
    fig = plt.figure(figsize=(12, 8)) # summarize history of loss
    plt.plot(X1[:, i], Y, 'bo')
    plt.ylabel('ConvexArea', fontsize=24)
    plt.xlabel('X'+str(i+1), fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(wd+'/ConvexArea vs X'+str(i+1)+'.png' , dpi=600)
    plt.close(fig)

### P3-2
xtrain, xtest = X1[0:600, :], X1[600:X1.shape[0], :]
ytrain, ytest = Y[0:600], Y[600:len(Y)]

Corr = np.corrcoef(np.transpose(np.concatenate((xtrain, np.reshape(ytrain, (len(ytrain), 1))), 1)))

df = pd.DataFrame(Corr)
df.to_csv(wd+'/A1-P3(2) Corr.csv', index=False)

### P3-3
reg = LinearRegression().fit(xtrain, ytrain)
reg.coef_

### P3-4
fitted = reg.predict(xtest)
MSE = np.mean((fitted-ytest)**2)

### P3-6
pca = PCA().fit(xtrain)
var = pca.explained_variance_ratio_
print(var)

### P3-7
df = pd.DataFrame({'variance': var})

temp = []
for i in range(len(var)):
   temp.append('PC'+str(i+1))
df.index = temp
   
df = df.sort_values(by='variance', ascending=False)
df["cumpercentage"] = df["variance"].cumsum()/df["variance"].sum()*100

fig, ax = plt.subplots()
ax.bar(df.index, df["variance"], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()

### P3-8
PC = pca.transform(xtrain)[:, 0:3]
PC_te = pca.transform(xtest)[:, 0:3]

reg = LinearRegression().fit(PC, ytrain)

fitted = reg.predict(PC_te)
MSE = np.mean((fitted-ytest)**2)
