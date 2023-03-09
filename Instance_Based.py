import os
import csv
import random

import scipy
from PIL import Image as PILImage
import numpy as np
import matplotlib.image as mpimage

from matplotlib.pyplot import figure
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import time as time
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model, metrics

size = 256 * 256
base_path = ["/Users/surabhigupta/Documents/Maths/Math251/Project/test",
             "/Users/surabhigupta/Documents/Maths/Math251/Project/train"]


def change_to_pixel(path, size):
    entries = os.listdir(path)
    entries = [x for x in entries if x.endswith(".png")]
    pixel = np.zeros((len(entries), size))
    for i in range(len(entries)):
        an_image = PILImage.open(os.path.join(path, entries[i]))
        pixel[i, :] = np.array(an_image.getdata())
    return pixel

train=change_to_pixel(base_path[1],size)    #later just need to open the save file-it helps save time

metaTrain = pd.read_csv(r'/Users/surabhigupta/Documents/Maths/Math251/Project/train.csv')

# test_data = pd.read_csv("/Users/gabati/Documents/Fall_22/Math251/Project/test.csv")
# data_entry = pd.read_csv("/Users/gabati/Documents/Fall_22/Math251/Project/Data_Entry_2017.csv")
# test_list = pd.read_csv("/Users/gabati/Documents/Fall_22/Math251/Project/test_list.txt", header=None)

Name_id = {0: 'Aortic enlargement', 1: 'Atelectasis', 2: 'Calcification', 3: 'Cardiomegaly',
           4: 'Consolidation', 5: 'ILD', 6: 'Infiltration', 7: 'Lung Opacity',
           8: 'Nodule/Mass', 9: 'Other lesion', 10: 'Pleural effusion', 11: 'Pleural thickening',
           12: 'Pneumothorax', 13: 'Pulmonary fibrosis', 14: 'No finding'}
Ytrain=[]
for i in range(len(files)):
    class_id=metaTrain.class_id[metaTrain.image_id.values==files[i]]
    Ytrain.append(Counter(class_id).most_common(1)[0][0])
files = os.listdir(base_path[1])
files = [file.replace(".png", "") for file in files]

####Splitting data
X_train,  X_test, y_train, y_test=train_test_split(Xtrain, Ytrain, test_size=0.2)
y_train=np.array(y_train)

###Normalize the data
X_norm=(X_train-np.mean(X_train, axis=0))/np.std(X_train, axis=0)
x_norm=(X_test-np.mean(X_train, axis=0))/np.std(X_train, axis=0)
################################ Plain Knn before converting to binary ############################


def k_nearest_neighbors_dist_idx(X_train, Y_train, x_test, k, metrics):
    neigh = NearestNeighbors(n_neighbors=k, metric=metrics)  # find the 13 nearest neighbors
    neigh.fit(X_train, Y_train)
    (dist, idx) = neigh.kneighbors(x_test, return_distance=True)  # get index of 13 nearest neighbors
    return (dist, idx)

kneighbors = list(range(1, 13))  # create a list of k that we want to perform knn

(dist,idx)=k_nearest_neighbors_dist_idx(X_norm, y_train, x_norm, 13,'euclidean') #get distance and indice matrix of 13 nearest neighbors

test_error_plain_15classes = []
for k in kneighbors:
    pred = []
    index = idx[:, :k]  # get the index of k nearest neighbors
    for i in range(len(idx)):
        target = y_train[index[i]]  # find the labels of the k nearest points
        pred.append(Counter(target).most_common(1)[0][0])
    test_error_plain_15classes.append(np.count_nonzero(y_test != pred) / len(x_norm))

print("Plain knn test error for different value of k with 15 classes: ",test_error_plain_15classes)



##Converting to binary class
y_train[y_train!=14]=0
y_train[y_train==14]=1
y_test=np.array(y_test)
y_test[y_test!=14]=0
y_test[y_test==14]=1



### Computing truncated dataset with 95% preserve variance
U, S, V_T = svd(X_norm, full_matrices=False)
s_matrix = np.multiply(S, S)  # element wise multiplication of S
fracs = np.cumsum(s_matrix / pow(LA.norm(S), 2))
k_95 = np.where(fracs >= 0.95)[0][1]  # gives us n_components = 349
k_80 = np.where(fracs >= 0.80)[0][1] # dim=21

##Define a red_data to obtain a reduced dataset from PCA
def red_data(X_norm, k, x_norm):
    u, s, vt = svds(X_norm, k)  # svd for lower
    X_trn_red = u @ np.diag(s)
    v = np.transpose(vt)
    X_tst_red = np.dot(x_norm, v)
    return X_trn_red, X_tst_red

(X_trn_red_95, X_tst_red_95) = red_data(X_norm, k_95, x_norm) 


################################ Plain Knn ############################

(dist,idx)=k_nearest_neighbors_dist_idx(X_norm, y_train, x_norm, 13,'euclidean') #get distance and indice matrix of 13 nearest neighbors

test_error_plain = []
for k in kneighbors:
    pred = []
    index = idx[:, :k]  # get the index of k nearest neighbors
    for i in range(len(idx)):
        target = y_train[index[i]]  # find the labels of the k nearest points
        pred.append(Counter(target).most_common(1)[0][0])
    test_error_plain.append(np.count_nonzero(y_test != pred) / len(x_norm))

print("Plain knn test error for different value of k: ",test_error_plain)

############################# Plain knn +95% PCA #########################


start=time.time()
(dist_red_95,idx_red_95)=k_nearest_neighbors_dist_idx(X_trn_red_95, y_train, X_tst_red_95, 13,'euclidean') #get distance and indice matrix of 13 nearest neighbors
test_error_knn_95 = []
F1_score_knn_95=[]
for k in kneighbors:
    pred = []
    index = idx_red_95[:, :k]  # get the index of k nearest neighbors
    for i in range(len(idx_red_95)):
        target = y_train[index[i]]  # find the labels of the k nearest points
        pred.append(Counter(target).most_common(1)[0][0])
    test_error_knn_95.append(np.count_nonzero(y_test != pred) / len(X_tst_red_95))
    F1_score_knn_95.append(metrics.f1_score(y_test,pred))
print("Plain knn +95% PCA test error for different value of k: ",test_error)
print("Plain knn +95% PCA F1 scores for different value of k: ", F1_score_knn_9)
end=time.time()
print(end-start)


######################### knn +pca +lda ###########################


n_features = X_trn_red_95.shape[1]
class_labels = np.unique(y_train)
# Within class scatter matrix:
# SW = sum((X_c - mean_X_c)^2 )
# Between class scatter:
# SB = sum( n_c * (mean_X_c - mean_overall)^2 )
mean_overall = np.mean(X_trn_red_95, axis=0)
SW = np.zeros((n_features, n_features))
SB = np.zeros((n_features, n_features))
# import pdb; pdb.set_trace()
for c in class_labels:
    X_c = X_trn_red_95[y_train == c]
    mean_c = np.mean(X_c, axis=0)
    # (4, n_c) * (n_c, 4) = (4,4) -> transpose
    SW += (X_c - mean_c).T.dot((X_c - mean_c))
    # (4, 1) * (1, 4) = (4,4) -> reshape
    n_c = X_c.shape[0]
    mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
    SB += n_c * (mean_diff).dot(mean_diff.T)
# Determine SW^-1 * SB
from sklearn import preprocessing

# Get eigenvalues and eigenvectors of SW^-1 * SB
E_lda, V_lda = scipy.linalg.eigh(SB, SW, eigvals_only=False)

indx = np.argsort(abs(E_lda))[::-1]
eigenVectors = V_lda[:,indx]

z_train = np.dot(X_trn_red_95, eigenVectors[:,:1])
z_test = np.dot(X_tst_red_95, eigenVectors[:,:1]) # LDA

(dist, idx) = k_nearest_neighbors_dist(z_train, y_train, z_test, 12)
error_pca_lda_knn = []
F1_score_pca_lda_knn[]
for k in range(1, 13):
    pred = []
    if k == 2:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(z_train, y_train)
        pred = knn.predict(z_test)
    else:
        index = idx[:, :k]
        for i in range(len(idx)):
            labels = y_train[index[i]]  # find the labels of the k nearest points
            pred.append(Counter(labels).most_common(1)[0][0])
    error_pca_lda_knn.append(np.count_nonzero(y_test != pred) / len(z_test))
    F1_score_pca_lda_knn.append(metrics.f1_score(pred, y_test))

print("pca+lda+knn test error for different value of k: ",error_pca_lda_knn)
print("pca+lda+knn f1 scores for different value of k: ",F1_score_pca_lda_knn)



################################## NLC #####################################

start=time.time()
groupX=[]  #create a list groupX to store the x_train of 10 different groups  len(groupX) is supposed to be 10
groupY=[]  #create a list groupY to store the y_train of 10 different groups
def k_nearest_neighbors_idx(X_train, Y_train, x_test, k, metrics):
    neigh=NearestNeighbors(n_neighbors=k, metric=metrics) #find the 13 nearest neighbors
    neigh.fit(X_train, Y_train)
    idx=neigh.kneighbors(x_test, return_distance=False) #get index of 13 nearest neighbors
    return idx
for i in range(2):
    group_Xi=X_train[np.where(y_train==i)]
    groupX.append(group_Xi)
    groupY.append(y_train[np.where(y_train==i)])
index_NLC=[]
test_error_NLC=[]
F1_score_NLC=[]
def dist_eu(x,y):    #dist function will calculate the distance (in euclidean) between two points x and y
    diff_square=[(x[i]-y[i])**2 for i in range(len(x))]
    dist=np.sqrt(sum(diff_square))
    return dist
for i in range(2):       #find the indices of 12 closest neighbors of x_test (for 10 different groups)
    idx_i=k_nearest_neighbors_idx(groupX[i], groupY[i], x_norm, 12, 'euclidean')
    index_NLC.append(idx_i)
for k in kneighbors:
    centroid_dist=np.zeros((len(x_norm),2))
    for i in range(2):
        X_coord_i=groupX[i][index_NLC[i][:,:10]]   #X_coord_i is the coordinate of the 12 nearest neighbors
        centroid_i=[np.mean(X_coord_i[j,:],axis=0) for j in range(len(X_test))] #centroid_i is the local centroids of group i
        dist_i=[dist_eu(centroid_i[10], X_test[10,:]) for k in range(len(X_test))] #dist_i is the distance from test point to local centroid of group i
        centroid_dist[:,i]=dist_i      #add the dist_i to column i in centroid_dist matrix
    pred_NLC=np.argmin(centroid_dist, axis=1)   #pred_NLC is the prediction of x_test
    test_error_NLC.append(np.count_nonzero(y_test!=pred_NLC)/len(y_test))
    F1_score_NLC.append(metrics.f1_score(y_test, pred_NLC))
    
print("NLC test error for different value of k: ",test_error_NLC)
print("NLC f1 scores for different value of k: ",F1_score_NLC)



###Visualization

# Display some image with multiple class
multilabel = metaTrain[['image_id', 'class_name', 'rad_id']][metaTrain.image_id.values == files[1]]

##Display some images

path = base_path[1]
entries = os.listdir(path)
entries = [x for x in entries if x.endswith(".png")]
Labels = Ytrain[:6]  # We don't need picture for predicted labels
row, col = (1, 3)
fig, axes = plt.subplots(row, col, figsize=(1.5 * col, 1.25 * row))
for j in range(3):
    ax = axes[j]
    j = j + 3
    an_image = mpimage.imread(os.path.join(path, entries[j]))
    imgploy = ax.imshow(an_image, cmap='gray')
    ax.set_title('{}:{}'.format(Labels[j], Name_id[Labels[j]]))
for axes in axes.flat:
    axes.label_outer()
plt.tight_layout()
plt.show()

##################### PCA with 2 or 3 components ###########
pca=PCA(n_components=3)
pca.fit(X_norm)
X_pca=pca.transform(X_norm)
ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)

print("KKKKKKKKKKK")
Xax = X_pca[:,0]
Yax = X_pca[:,1]
Zax = X_pca[:,2]

cdict = {0:'red',1:'green'}
labl = {0:'Un-healthy',1:'Healthy'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')
for l in np.unique(y_train):
 ix=np.where(y_train==l)
 ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
           label=labl[l], marker=marker[l], alpha=alpha[l])
# for loop ends
ax.set_xlabel("1st PC", fontsize=14)
ax.set_ylabel("2nd PC", fontsize=14)
ax.set_zlabel("3rd PC", fontsize=14)

ax.legend()
plt.show()
#
#
#
Xax=X_pca[:,0]
Yax=X_pca[:,1]
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(y_train):
 ix=np.where(y_train==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l],marker=marker[l],alpha=alpha[l])
# for loop ends
plt.xlabel("1st PC",fontsize=14)
plt.ylabel("2nd PC",fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize=(15,10))
lw = 2
sc = plt.scatter(X_trn_red_95.T[0], X_trn_red_95.T[1], c=y_train, cmap=plt.cm.coolwarm, alpha=1, lw=lw)
clb = plt.colorbar(sc)
clb.ax.set_title('Class', fontsize=15)
plt.xlabel("1st PC", fontsize=12)
plt.ylabel("2nd PC", fontsize=12)
plt.title('2D PCA of Chest Xray dataset', fontweight = 'bold', fontsize=15)

plt.show()


####Display errors for instance-based classifiers
plt.plot(kneighbors, test_error_plain,'-', label="Plain kNN with binary")
plt.plot(kneighbors, test_error_plain_15classes,'-.', label="Plain kNN with 15 classes")
plt.legend()
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.title('kNN Error rates before and after converting to binary')
plt.show()



plt.plot(kneighbors, test_error_plain,'-', label="Plain kNN")
plt.plot(kneighbors, test_error_knn_95,'-.', label="PCA 95% + kNN")
plt.plot(kneighbors, error_pca_lda_knn,'--*', label="PCA 95% +LDA + kNN")
plt.plot(kneighbors,test_error_NLC,'--p', label="NLC")
plt.legend()
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.title('Error rates between different instance-based classifiers')
plt.show()