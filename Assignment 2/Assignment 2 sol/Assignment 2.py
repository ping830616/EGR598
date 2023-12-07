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
import cvlib as cv
import os, csv, cv2, math, itertools, random
from cvlib.object_detection import draw_bbox
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

wd = 'C:/Users/sheng/Downloads/class (teaching)/EGR 598 ML & AI/Assignment' # directory to save results

''' Part 1 '''
C = [20, 40, 60]
K = [4, 8, 16]

image = cv2.imread(wd+'/Assignment 2_image.jpg')

### P1-1 and P1-3
RE1, RE2 = np.empty((len(K), len(C))), np.empty((len(K), len(C)))
for k in K:
    for c in C:
        run1, run2 = int(image.shape[0]/c), int(image.shape[1]/c) # number of windows
        
        sample = []
        for i1 in range(run1):
            for i2 in range(run2):
                img = image[i1*c:(i1+1)*c, i2*c:(i2+1)*c, :].flatten('F') # default order of flattening is by 'row'
                sample.append(img)
        sample = np.array(sample)
        
        # k-means clustering
        fit1 = sc.KMeans(n_clusters=k).fit(sample)  
        lbs1 = fit1.labels_
        refs1 = fit1.cluster_centers_.astype('int')

        # agglomerrative clsutering
        fit2 = sc.AgglomerativeClustering(linkage='complete', n_clusters=k).fit(sample)  
        lbs2 = fit2.labels_
        refs2 = []
        for i in range(k):
            refs2.append(np.mean(sample[np.where(lbs2==i)[0], :], 0))
        refs2 = np.array(refs2).astype('int')

        sample1, sample2 = [], []
        image1 = np.empty((image.shape[0], image.shape[1], 3)).astype('int') # make sure that you use "integer" data type for pixel values
        image2 = np.empty((image.shape[0], image.shape[1], 3)).astype('int')
        for i1 in range(run1):
            for i2 in range(run2):
                # k-means
                img1 = refs1[lbs1[i1*run1+i2]]
                sample1.append(img1)
                
                img1a = np.empty((c, c, 3)).astype('int')
                for j1 in range(3):
                    temp = img1[j1*c**2:(j1+1)*c**2]
                    for j2 in range(c):
                        img1a[:, j2, j1] = temp[j2*c:(j2+1)*c]
                    
                image1[i1*c:(i1+1)*c, i2*c:(i2+1)*c, :] = img1a
                
                # Agglomerative
                img2 = refs2[lbs2[i1*run1+i2]]
                sample2.append(img2)
                
                img2a = np.empty((c, c, 3)).astype('int')
                for j1 in range(3):
                    temp = img2[j1*c**2:(j1+1)*c**2]
                    for j2 in range(c):
                        img2a[:, j2, j1] = temp[j2*c:(j2+1)*c]
                    
                image2[i1*c:(i1+1)*c, i2*c:(i2+1)*c, :] = img2a
          
        # k-means
        fig = plt.figure(figsize=(10, 10)) 
        plt.imshow(image1)
        plt.axis('off')
        plt.savefig(wd+'/A2-1(1) kmeans c'+str(c)+' k'+str(k)+'.png' , dpi=600)
        plt.close(fig)
        
        # Agglomerative
        fig = plt.figure(figsize=(10, 10)) 
        plt.imshow(image2)
        plt.axis('off')
        plt.savefig(wd+'/A2-1(3) agglomerative c'+str(c)+' k'+str(k)+'.png' , dpi=600)
        plt.close(fig)
        
        RE1[K.index(k), C.index(c)] = np.mean((sample-np.array(sample1))**2)
        RE2[K.index(k), C.index(c)] = np.mean((sample-np.array(sample2))**2)

print(RE1), print('\n') # reconstruction error - row for K and column for C
print(RE2), print('\n') # RE1 for k-means and RE2 for Agglomerrative

''' Part 2 '''
data = pd.read_csv(wd+'/iris.data', sep=",")
data = pd.DataFrame(data).to_numpy()

Cls = data[:, 4]
X = data[:, 0:4].astype('float')

### P2-1
h = [1, 0.5, 0.25]

def GaussianK(x, X, h): # function for Gaussian kernel estimation
    temp = multivariate_normal.pdf((x-X)/h, mean=np.zeros(4), cov=np.eye(4)) # use a MVN density function as the Gaussian kernel
    return np.sum(temp)/(X.shape[0]*h)

X1 = X[np.where(Cls=='Iris-setosa')[0], :]

x = np.empty((X1.shape[0], X1.shape[1]))
for i in range(X1.shape[1]):
    delta = (np.max(X1[:, i])-np.min(X1[:, i]))/X1.shape[0]
    x[:, i] = np.arange(np.min(X1[:, i]), np.max(X1[:, i]), delta)[0:X1.shape[0]]

for i in h:
    E = []
    for k in x:
        E.append(GaussianK(k, X1, i))
    
    for j in range(X1.shape[1]):
        fig = plt.figure(figsize=(10, 10)) 
        plt.plot(x[:, j], E)
        plt.ylabel('Kernel Estimate', fontsize=24)
        plt.xlabel('X'+str(j+1), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(wd+'/A2-2(1) h'+str(i)+' variable'+str(j+1)+'.png' , dpi=600)
        plt.close(fig)

### P2-2
p = 0.5
Set = set(Cls)

xtrain, xtest = np.empty((0, X.shape[1])).astype('float'), np.empty((0, X.shape[1])).astype('float')
ytrain, ytest = np.empty((0, 1)), np.empty((0, 1))
for i in range(len(Set)):
    cl = Set.pop()
    
    temp = np.where(Cls==cl)[0]
    xtrain, xtest = np.append(xtrain, X[temp[0:int(p*len(temp))], :], 0), np.append(xtest, X[temp[int(p*len(temp)):len(temp)], :], 0)
    ytrain, ytest = np.append(ytrain, Cls[temp[0:int(p*len(temp))]]), np.append(ytest, Cls[temp[int(p*len(temp)):len(temp)]])

with open(wd+'/A2-P2(2) training.csv', 'w', newline='') as csvfile: 
    writer = csv.writer(csvfile)  
    writer.writerows(np.concatenate((xtrain, np.reshape(ytrain, (len(ytrain), 1))), 1))

with open(wd+'/A2-P2(2) testing.csv', 'w', newline='') as csvfile: 
    writer = csv.writer(csvfile)  
    writer.writerows(np.concatenate((xtest, np.reshape(ytest, (len(ytest), 1))), 1))

### P2-3
K = [3, 5, 10, 20]

for k in K:
    nbrs = KNeighborsClassifier(n_neighbors=k)
    nbrs.fit(xtrain, ytrain)
    
    ypred = nbrs.predict(xtest)
    
    M1 = confusion_matrix(ytest, ypred, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) # confusion matrix for 2
    print(M1) 
    print(np.trace(M1)/np.sum(M1))
    
### P2-4
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(xtrain, ytrain)  

tree.plot_tree(clf) # visualize the tree structure   
                
ypred = clf.predict(xtest)
    
M1 = confusion_matrix(ytest, ypred, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) # confusion matrix for 2
print(M1) 
print(np.trace(M1)/np.sum(M1))

### P2-5
clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)                  
ypred = clf.predict(xtest)
    
M1 = confusion_matrix(ytest, ypred, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) # confusion matrix for 2
print(M1) 
print(np.trace(M1)/np.sum(M1))

''' Part 3 '''
image = cv2.imread(wd+'/Assignment 2_image 1.jpg')

def get_output_layers(net):
   layer_names = net.getLayerNames()
   output_layers = net.getUnconnectedOutLayersNames()
   return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
   label = str(classes[class_id])
   
   color = COLORS[class_id]

   cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
   cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(wd+'/yolov3.txt', 'r') as f:
   classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(wd+'/yolov3.weights',wd+'/yolov3.cfg.txt')
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)

outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
   for detection in out:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if confidence > 0.5:
        center_x = int(detection[0] * Width)
        center_y = int(detection[1] * Height)
        w = int(detection[2] * Width)
        h = int(detection[3] * Height)
        x = center_x - w / 2
        y = center_y - h / 2
        class_ids.append(class_id)
        confidences.append(float(confidence))
        boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
   box = boxes[i[0]]
   x = box[0]
   y = box[1]
   w = box[2]
   h = box[3]
   draw_prediction(image, class_ids[i[0]], confidences[i[0]], round(x), round(y), round(x+w), round(y+h))

cv2.imshow(wd+"/object detection", image)
cv2.waitKey()

cv2.imwrite(wd+"/object-detection.jpg", image)
cv2.destroyAllWindows()