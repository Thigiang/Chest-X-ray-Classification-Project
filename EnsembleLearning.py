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



################# Random forest for tree 50 to 500  for depth 50   ############
oob_score_dif_max_depth=[]
for m in [10,20,50]:
    start_time = time.time()
    n_estimators = [i for i in range(0, 501, 50)]
    n_estimators[0]=10
    min_sam_leaf = [1,3,5,7]
    accuracy = []
    oob_sc = []
    for i in min_sam_leaf:
        for j in n_estimators:
            rf = RandomForestClassifier(n_estimators=j, max_depth=m ,oob_score= True,min_samples_leaf=i)
            rf.fit(X_trn_red_95, y_train)
            y_pred = rf.predict(X_trn_red_95)
            accuracy.append(accuracy_score(y_test,y_pred))
            oob_sc.append(rf.oob_score_)
    end_time = time.time()
    total_time = end_time - start_time
    oob_score_dif_max_depth.append(oob_sc)
    print("Total time taken to run time for depth {}: ".format(m) , total_time)
    print("Accuracy for Random forest for tree 10 to 500  for depth {}: ".format(m),accuracy)
    print("oob for Random forest for tree 50 to 100  for depth {}: ".format(m), oob_sc)

print("OOB score for all combinations: ", oob_score_dif_max_depth)



################# Random forest for tree 100 with depth 20  ################


start_time = time.time()
accuracy_RF_20 = []
rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=7, max_depth=20)
rf.fit(X_norm, y_train)
y_pred = rf.predict(x_norm)
accuracy_RF_20.append(accuracy_score(y_test,y_pred))
print("accuracy with depth 20 and tree = 100:  ",accuracy_RF_20)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



# ####################### Ada boosting ################################
# #
# print("VVVVVVVVVVVVVVVV")
# start_time = time.time()
# dc_tree = DecisionTreeClassifier(max_depth = 20)
# accuracy = []
# clf = AdaBoostClassifier(base_estimator=dc_tree,n_estimators=50, random_state=0)
# clf.fit(X_trn_red_95, y_train)
# y_pred = clf.predict(X_tst_red_95)
# accuracy.append(accuracy_score(y_test,y_pred))
# end_time = time.time()
# total_time = end_time - start_time
# print("Accuracy for max depth 20, T = 50, AdaBoosting: ", accuracy)
# print("Total time is: ", total_time)



start_time = time.time()
dc_tree = DecisionTreeClassifier(max_depth = 20)
accuracy_Adabo_20 = []
clf = AdaBoostClassifier(base_estimator=dc_tree,n_estimators=100, random_state=0)
clf.fit(X_trn_red_95, y_train)
y_pred = clf.predict(X_tst_red_95)
accuracy_Adabo_20.append(accuracy_score(y_test,y_pred))
end_time = time.time()
total_time = end_time - start_time
print("Accuracy for max depth 20, T = 100, AdaBoosting: ", accuracy_Adabo_20)
print("Total time is: ", total_time)

############################## gradient Boosting ########################

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model, metrics
learning_rate=[0.1,0.3,0.5,1,2]
accuracyGB_logloss=[]
for lr in learning_rate:
    start=time.time()
    GB=GradientBoostingClassifier(max_depth=20, n_estimators=100, learning_rate=lr).fit(X_norm,y_train)
    pred=GB.predict(x_norm)
    accuracyGB=metrics.accuracy_score(y_test, pred)
    accuracyGB_logloss.append(accuracyGB)
    end=time.time()
    print("Time and accuracy for learning rate = {}:".format(0.1),end-start,",",accuracyGB)
print("accuracy rate for gradient boosting with log loss",accuracyGB_logloss)

accuracyGB_exponential=[]
for lr in learning_rate:
    start=time.time()
    GB=GradientBoostingClassifier(max_depth=20, n_estimators=100, learning_rate=lr,loss='exponential' ).fit(X_trn_red_95,y_train)
    pred=GB.predict(X_tst_red_95)
    accuracyGB=metrics.accuracy_score(y_test, pred)
    accuracyGB_exponential.append(accuracyGB)
    end=time.time()
    print("Time and accuracy for learning rate = {}:".format(lr),end-start,",",accuracyGB)

print("accuracy rate for gradient boosting with exponential loss",accuracyGB_exponential)

accuracyGB_deviance=[]
for lr in learning_rate:
    start=time.time()
    GB=GradientBoostingClassifier(max_depth=20, n_estimators=100, learning_rate=lr,loss='deviance' ).fit(X_trn_red_95,y_train)
    pred=GB.predict(X_tst_red_95)
    accuracyGB=metrics.accuracy_score(y_test, pred)
    accuracyGB_deviance.append(accuracyGB)
    end=time.time()
    print("Time and accuracy(Deviance) for learning rate = {}:".format(lr),end-start,",",accuracyGB)

print("accuracy rate for gradient boosting with deviance loss",accuracyGB_deviance)


GB=GradientBoostingClassifier(max_depth=20, n_estimators=100, learning_rate=2,loss='exponential' ).fit(X_trn_red_95,y_train)
pred=GB.predict(X_tst_red_95)
accuracyGB_20=metrics.accuracy_score(y_test, pred)


################### VISUALIZATION ##############

#### OOB error ###############
Trees = [i for i in range(0, 501, 50)]
Trees[0]=10
oob_score_dif_max_depth

###max_depth =10###
plt.plot(Trees, oob_score_dif_max_depth[0][0],'-', label="minleafsize=1")
plt.plot(Trees, oob_score_dif_max_depth[0][1],'-.', label="minleafsize= 3")
plt.plot(Trees, oob_score_dif_max_depth[0][2],'--*', label="minleafsize=5")
plt.plot(Trees, oob_score_dif_max_depth[0][3],'--*', label="minleafsize=7")
plt.legend()
plt.xlabel('# of grown tree, T')
plt.ylabel('Error Rate')
plt.title('OOB error for differnt trees with max_depth=10')
plt.show()

###max_depth =20###
plt.plot(Trees, oob_score_dif_max_depth[1][0],'-', label="minleafsize=1")
plt.plot(Trees, oob_score_dif_max_depth[1][1],'-.', label="minleafsize= 3")
plt.plot(Trees, oob_score_dif_max_depth[1][2],'--*', label="minleafsize=5")
plt.plot(Trees, oob_score_dif_max_depth[1][3],'--*', label="minleafsize=7")
plt.legend()
plt.xlabel('# of grown tree, T')
plt.ylabel('Error Rate')
plt.title('OOB error for differnt trees with max_depth=20')
plt.show()

###max_depth =50###
plt.plot(Trees, oob_score_dif_max_depth[2][0],'-', label="minleafsize=1")
plt.plot(Trees, oob_score_dif_max_depth[2][1],'-.', label="minleafsize= 3")
plt.plot(Trees, oob_score_dif_max_depth[2][2],'--*', label="minleafsize=5")
plt.plot(Trees, oob_score_dif_max_depth[2][3],'--*', label="minleafsize=7")
plt.legend()
plt.xlabel('# of grown tree, T')
plt.ylabel('Error Rate')
plt.title('OOB error for differnt trees with max_depth=50')
plt.show()




plt.plot(learning_rate, accuracyGB_logloss,'-', label="Log-loss")
plt.plot(learning_rate,accuracyGB_exponential,'-.', label="Exponential loss")
plt.plot(learning_rate, accuracyGB_deviance,'--*', label="Deviance loss")
plt.xticks(range(len(learning_rate)), learning_rate)
plt.legend()
plt.xlabel('Learning rate')
plt.ylabel('Accuracy Rate')
plt.title('Accuracy rate for gradient boosting with differnt learning rates')
plt.show()



###Bar plot for 3 different classifiers
plt.bar(range(3),accuracy_RF_20)
plt.bar(range(3),accuracy_Adabo_20)
plt.bar(range(3),accuracyGB_20)
plt.xticks(range(3),['Random Forest','Adaboosting','Gradient Boosting'])
plt.title('Accuracy rate for differnt tree classifiers')
plt.xlabel('Methods')
plt.ylabel('Accuracy Rate')
plt.show()

