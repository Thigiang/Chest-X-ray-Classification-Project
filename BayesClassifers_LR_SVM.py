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


############################## Bayes Classifier ####################
###Plot accuracy rates for Bayes classifiers####
plt.plot(kneighbors, accuracy_lda,'-', label="LDA")
plt.plot(kneighbors, accuracy_qda,'-.', label="QDA")
plt.plot(kneighbors, accuracy_gnb,'--*', label="GNB")
plt.legend()
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.title('Error rates between different instance-based classifiers')
plt.show()

accuracy_lda = []
F1_score_lda=[]
accuracy_qda = []
F1_score_qda=[]
accuracy_gnb = []
F1_score_gnb=[]
for i in range(k_80,k_95+1,20):
    X_trn_red_i, X_tst_red_i = red_data(X_norm, i, x_norm)
    lda = LinearDiscriminantAnalysis()
    model_lda = lda.fit(X_trn_red_i, y_train)
    pred_lda = model_lda.predict(X_tst_red_i)
    accuracy_lda.append(accuracy_score(y_test, pred_lda))
    F1_score_lda.append(metrics.f1_score(y_test,pred_lda))

    qda = QuadraticDiscriminantAnalysis()
    model_qda = qda.fit(X_trn_red_i, y_train)
    pred_qda = model_qda.predict(X_tst_red_i)
    accuracy_qda.append(accuracy_score(y_test, pred_qda))
    F1_score_qda.append(metrics.f1_score(y_test,pred_qda))


    clf = GaussianNB()
    model_gnb = clf.fit(X_trn_red_i, y_train)
    pred_gnb = model_gnb.predict(X_tst_red_i)
    accuracy_gnb.append(accuracy_score(y_test, pred_gnb))
    F1_score_gnb.append(metrics.f1_score(y_test,pred_gnb))


###Bayes on original data

lda = LinearDiscriminantAnalysis()
model_lda = lda.fit(X_train, y_train)
pred_lda = model_lda.predict(X_test)
accuracy_lda.append(accuracy_score(y_test, pred_lda))
F1_score_lda.append(metrics.f1_score(y_test,pred_lda))

qda = QuadraticDiscriminantAnalysis()
model_qda = qda.fit(X_train, y_train)
pred_qda = model_qda.predict(X_test)
accuracy_qda.append(accuracy_score(y_test, pred_qda))
F1_score_qda.append(metrics.f1_score(y_test,pred_qda))


clf = GaussianNB()
model_gnb = clf.fit(X_train, y_train)
pred_gnb = model_gnb.predict(X_test)
accuracy_gnb.append(accuracy_score(y_test, pred_gnb))
F1_score_gnb.append(metrics.f1_score(y_test,pred_gnb))


print("accuracy_lda", accuracy_lda)
print("accuracy_qda", accuracy_qda)
print("accuracy_gnb", accuracy_gnb)
print('F1 score lda',F1_score_lda)
print('F1 score qda', F1_score_qda)
print('F1 score gnb', F1_score_gnb)

end=time.time()
print(end-start)






######################### Logistic regression #######################

# # One Vs Rest after PCA
c = []
for i in range(-4, 6):
    print(i)
    c.append(pow(2, i))

error_LR = []
accuracy_LR = []
total_time = []
F1_LR=[]
for i in c:
    start_time = time.time()
    logreg_ovr = linear_model.LogisticRegression(max_iter=10000, C = i)
    logreg_ovr.fit(X_train, y_train)
    pred_ovr = logreg_ovr.predict(X_test)
    accuracy_LR.append(logreg_ovr.score(X_test, y_test))
    F1_LR.append(metrics.f1_score(y_test,pred_ovr))
    error_LR.append(1 - accuracy[-1])
    end_time = time.time()
    total_time.append(end_time - start_time)
print("Total time taken to run ovr Logistic regression is: ", total_time)
print(f"Logistic Regression Model's accuracy for each value of c: {accuracy_LR}")
print(f"Logistic Regression Model's error for each value of c: {error_LR}")
print('F1 for LR', F1_LR_pca)


error_LR_pca = []
accuracy_LR_pca = []
total_time = []
F1_LR_pca=[]
for i in c:
    start_time = time.time()
    logreg_ovr = linear_model.LogisticRegression(max_iter=10000, C = i)
    logreg_ovr.fit(X_trn_red_95, y_train)
    pred_ovr = logreg_ovr.predict(X_tst_red_95)
    accuracy_LR_pca.append(logreg_ovr.score(X_tst_red_95, y_test))
    F1_LR_pca.append(metrics.f1_score(y_test,pred_ovr))
    error_LR_pca.append(1 - accuracy[-1])
    end_time = time.time()
    total_time.append(end_time - start_time)
print("Total time taken to run ovr Logistic regression with pca is: ", total_time)
print(f"Logistic Regression Model's accuracy with pca for each value of c: {accuracy_LR_pca}")
print(f"Logistic Regression Model's error with pca for each value of c: {error_LR_pca}")
print('F1 for LR', F1_LR_pca)

#############################       SVM        #############################

# Initialize C-Support Vector classifier
c = []
for i in range(-4, 6):
    c.append(pow(2, i))
error_linear_svm = []
accuracy_linear_svm = []
total_time = []

for i in c:
    start_time = time.time()
    SVM = LinearSVC(C = i, dual=False)
    SVM.fit(X_trn_red_95, y_train)
    svm_pred = SVM.predict(X_tst_red_95)
    # Print accuracy on test data and labels
    accuracy_linear_svm.append(SVM.score(X_tst_red_95, y_test))
    error_linear_svm.append(1- accuracy[-1])
    end_time = time.time()
    total_time.append(end_time - start_time)

print("Total time taken to run ovo SVM is: ", total_time)
print(f"SVM Model's accuracy for each value of c: {accuracy_linear_svm}")
print(f"SVM Model's error for each value of c: {error_linear_svm}")




################### SVC with Kerenel polynomial and degree 3 ######################

error_poly_svm = []
accuracy_poly_svm = []
total_time = []
for i in c:
    start_time = time.time()
    SVM = SVC(C = i,decision_function_shape='ovo', kernel = 'poly', degree=3)
    # Fit classifier
    SVM.fit(X_trn_red_95, y_train)
    # Predict labels according
    Y_pred = SVM.predict(X_tst_red_95)
    # Print accuracy on test data and labels
    accuracy_poly_svm.append(SVM.score(X_tst_red_95, y_test))
    error_poly_svm.append(1- accuracy[-1])
    end_time = time.time()
    total_time.append(end_time - start_time)
print("Total time taken to run ovo SVM for poly kernel is: %.4f",  total_time)
print(f"for poly kernel SVM Model's accuracy for each value of c: {accuracy_poly_svm}")
print(f"for poly kernel SVM Model's error for each value of c: {error_poly_svm}")

######################### SVM with sigma and gamma #####################


subset = []
for i in range(100):
    number = random.randint(1, 12000)
    if number not in subset:
        subset.append(number)
n = len(subset)
X_sample_100 = X_trn_red_95[subset]
Y_sample_100 = y_train[subset]
sigma = 0
for (index, value) in enumerate(X_sample_100):
    label = Y_sample_100[index]
    X = X_trn_red_95[y_train == label]
    nbrs = NearestNeighbors(n_neighbors=7).fit(X) # value of  k =7
    dist, nb = nbrs.kneighbors([value])
    sigma += dist[0, 7-1]
sigma = sigma/n
gamma=1/(2*sigma**2)
print("Value of sigma: ",sigma)

error_gau_svm = []
accuracy_gau_svm = []
total_time = []
for i in c:
    start_time = time.time()
    SVM = SVC(C = i,decision_function_shape='ovo',kernel='rbf',gamma=gamma)
    # Fit classifier
    SVM.fit(X_trn_red_95, y_train)
    # Predict labels according
    Y_pred = SVM.predict(X_tst_red_95)
    # Print accuracy on test data and labels
    accuracy_gau_svm.append(SVM.score(X_tst_red_95, y_test))
    error_gau_svm.append(1- accuracy[-1])
    end_time = time.time()
    total_time.append(end_time - start_time)


print("Total time taken to run ovo SVM is: %.4f",  total_time)
print(f"SVM Model's accuracy for each value of c: {accuracy_gau_svm}")
print(f"SVM Model's error for each value of c: {error_gau_svm}")




######### Visualization #############

###accuracy rates for Bayes visualization ######
dimensions=list(range(k_80,k_95+1,20))
plt.plot(len(dimensions)+1, accuracy_lda,'-', label="LDA")
plt.plot(len(dimensions)+1, accuracy_qda,'-.', label="QDA")
plt.plot(len(dimensions)+1, accuracy_gnb,'--*', label="Gaussian Naive Bayes")
plt.xticks(len(dimensions), 'No PCA')
plt.legend()
plt.xlabel('Dimensions')
plt.ylabel('Accuracy Rate')
plt.title('Accuracy rates between different Bayes classifiers')
plt.show()



##### Accuracy rates for LR before vs after PCA ####
plt.plot(c, accuracy_LR,'-', label="Logistic Regression without PCA")
plt.plot(c, accuracy_LR_pca,'-.', label="Logistic Regression with PCA 95%")
plt.xticks(len(dimensions), 'No PCA')
plt.legend()
plt.xlabel('C')
plt.ylabel('Accuracy Rate')
plt.title('Accuracy rates for Logistic Regression with different C')
plt.show()
error_LR_pca = []
accuracy_LR_pca = []
###################### ROC-AUC for svm ##########################

plt.plot(c, accuracy_linear_svm,'-', label="Linear SVM")
plt.plot(c, accuracy_poly_svm,'-.', label="SVM Polynomial with degree 3")
plt.plot(c, accuracy_gau_svm,'--*', label="Gaussain Kernel SVM, sigma=155.18")
plt.xticks(len(dimensions), 'No PCA')
plt.legend()
plt.xlabel('Dimensions')
plt.ylabel('Accuracy Rate')
plt.title('Accuracy rates for SVMs with different C')
plt.show()


sigma = 155.18
gamma=1/(2*sigma**2)

SVM = SVC(C = 2,decision_function_shape='ovo',kernel='rbf',gamma=gamma, probability=True)
SVM.fit(X_trn_red_95,y_train)
# predict the labels on the validation dataset
predictions_SVM = SVM.predict(X_tst_red_95)
# Use the accuracy_score function to get the accuracy
print("F1 score for svm: ", metrics.f1_score(predictions_SVM, y_test))
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test))

probs = SVM.predict_proba(X_tst_red_95)
y_pred_proba = probs[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()




