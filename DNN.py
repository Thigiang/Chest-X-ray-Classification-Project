###################### Neural Network with 1 layer for logistics activation ################

start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(100),solver='adam',
                            learning_rate_init=lr,activation='logistic',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 1 layers(100), epoch 100, logistic:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(300),solver='adam',
                            learning_rate_init=lr,activation='logistic',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 1 layers(300), epoch 100, logistic:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)

start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(500),solver='adam',
                            learning_rate_init=lr,activation='logistic',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 1 layers(500), epoch 100, logistic:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)


############################ ROC-AUC curve for neural network #########################

start_time = time.time()
accuracy = []

clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(100),solver='adam',
                    learning_rate_init=0.001,activation='logistic',batch_size=500)
clf.fit(X_trn_red_95, y_train)
y_pred = clf.predict(X_tst_red_95)
print("F1_score for NN with 1 layers(100), epoch 100, logistic: ",metrics.f1_score(y_pred, y_test))

# Compute ROC curve and ROC area for each class
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# The process of drawing a roc-auc curve belonging to a specific class

plt.figure()
lw = 2 # line_width
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1]) # Drawing Curve according to 2. class
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()



###################### Neural Network with 1 layer for Relu activation ################

start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(100),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 1 layers(100), epoch 100, relu:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(300),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 1 layers(300), epoch 100, relu:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)

start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(500),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 1 layers(500), epoch 100, relu:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



################################## Neural Network for 2 hidden layers #########################

start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(400,200),solver='adam',
                            learning_rate_init=lr,activation='logistic',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 2 layers (400,200) epoch 100, logistic: ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)


start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(400,200),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 2layer (400,200) epoch 100, relu: ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(500,300),solver='adam',
                            learning_rate_init=lr,activation='logistic',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 2 layers, epoch 100, logistic:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)


start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(500,300),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 2 layers, epoch 100, relu:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



################################# Neural Network for 3 hidden layers #########################

start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=500,hidden_layer_sizes=(500,300,100),solver='adam',
                            learning_rate_init=lr,activation='logistic',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 3 layers, epoch 500, logistic: ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)




start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=100,hidden_layer_sizes=(500,300,100),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 3 layers and epoch 100, relu:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)



start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=300,hidden_layer_sizes=(500,300,100),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy with 3 hidden layers with epoch 300, relu:  ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)


start_time = time.time()
accuracy = []
for mini in [100,200,500]:
    a = []
    for lr in [0.0001,0.0005, 0.001, 0.005, 0.01]:
        clf = MLPClassifier(random_state=1, max_iter=500,hidden_layer_sizes=(500,300,100),solver='adam',
                            learning_rate_init=lr,activation='relu',batch_size=mini)
        clf.fit(X_trn_red_95, y_train)
        y_pred = clf.predict(X_tst_red_95)
        a.append(accuracy_score(y_test, y_pred))
    accuracy.append(a)
    print(accuracy)
print("Accuracy 3 hidden layers with epoch 500, relu: ",accuracy)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken to run is: " , total_time)





#########VISUALIZATION ###########
#################### Neural Network For 1 hidden layer ####################

Accuracy = [[0.89, 0.896, 0.9, 0.899, 0.882], [0.899, 0.898, 0.896, 0.901, 0.887], [0.896, 0.899, 0.899, 0.9, 0.899]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 1 hidden layers of sizes 100 with logistic activation')
plt.show()

Accuracy = [[0.9, 0.896, 0.9006666666666666, 0.897, 0.8776666666666667], [0.8953333333333333, 0.9003333333333333, 0.8986666666666666, 0.8933333333333333, 0.8846666666666667], [0.8923333333333333, 0.8963333333333333, 0.899, 0.901, 0.899]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 1 hidden layers of sizes 300 with logistic activation')
plt.show()

Accuracy = [[0.8996666666666666, 0.9, 0.9043333333333333, 0.8833333333333333, 0.876], [0.8986666666666666, 0.9, 0.902, 0.8946666666666667, 0.8833333333333333], [0.8936666666666667, 0.8993333333333333, 0.8993333333333333, 0.902, 0.8936666666666667]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 1 hidden layers of sizes 500 with logistic activation')
plt.show()


Accuracy = [[0.89, 0.896, 0.9, 0.899, 0.882], [0.899, 0.898, 0.896, 0.901, 0.887], [0.896, 0.899, 0.899, 0.9, 0.899]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 1 hidden layers of sizes 100 with relu activation')
plt.show()

Accuracy = [[0.9, 0.896, 0.9006666666666666, 0.897, 0.8776666666666667], [0.8953333333333333, 0.9003333333333333, 0.8986666666666666, 0.8933333333333333, 0.8846666666666667], [0.8923333333333333, 0.8963333333333333, 0.899, 0.901, 0.899]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 1 hidden layers of sizes 300 with relu activation')
plt.show()

Accuracy = [[0.8996666666666666, 0.9, 0.9043333333333333, 0.8833333333333333, 0.876], [0.8986666666666666, 0.9, 0.902, 0.8946666666666667, 0.8833333333333333], [0.8936666666666667, 0.8993333333333333, 0.8993333333333333, 0.902, 0.8936666666666667]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 1 hidden layers of sizes 500 with relu activation')
plt.show()

#################### Neural Network For 2 hidden layer ####################

Accuracy = [[0.897, 0.895, 0.904, 0.888, 0.891], [0.894, 0.898, 0.903, 0.899, 0.896], [0.888, 0.897, 0.8983333333333333, 0.8983333333333333, 0.9023333333333333]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 2 hidden layers of sizes (500,300) with logistic activation')
plt.show()


Accuracy = [[0.897, 0.895, 0.904, 0.888, 0.891], [0.894, 0.898, 0.903, 0.899, 0.896], [0.888, 0.897, 0.8983333333333333, 0.8983333333333333, 0.9023333333333333]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 2 hidden layers of sizes (400,200) with logistic activation')
plt.show()


Accuracy = [[0.897, 0.895, 0.904, 0.888, 0.891], [0.894, 0.898, 0.903, 0.899, 0.896], [0.888, 0.897, 0.8983333333333333, 0.8983333333333333, 0.9023333333333333]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 2 hidden layers of sizes (500,300) with relu activation')
plt.show()


Accuracy = [[0.897, 0.895, 0.904, 0.888, 0.891], [0.894, 0.898, 0.903, 0.899, 0.896], [0.888, 0.897, 0.8983333333333333, 0.8983333333333333, 0.9023333333333333]]
lr = [0.0001,0.0005, 0.001, 0.005, 0.01]
plt.plot(lr, Accuracy[0], color='r', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[1], color='blue', linestyle='dashed', marker='.', markersize=10)
plt.plot(lr, Accuracy[2], color='green', linestyle='dashed', marker='.', markersize=10)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for 100 epochs and 2 hidden layers of sizes (400,200) with relu activation')
plt.show()

#################### Neural Network For 3 hidden layer ####################

learningrate = [0.0001,0.0005, 0.001, 0.005, 0.01]
accuracy = [[0.90, 0.9, 0.9, 0.8856666666666667, 0.8733333333333333], [0.8956666666666667, 0.8956666666666667, 0.8806666666666667, 0.8853333333333333, 0.8806666666666667], [0.8966666666666666, 0.8996666666666666, 0.8983333333333333, 0.8853333333333333, 0.873]]
plt.plot(range(len(learningrate)),accuracy[0],'--*')
plt.plot(range(len(learningrate)),accuracy[1],'--*')
plt.plot(range(len(learningrate)),accuracy[2],'--*')
plt.xticks(range(len(learningrate)), learningrate,fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning rate',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy for 100 Epochs, 3 hidden layers of sizes(500,300,100) with logistic',fontsize=14)
plt.show()



learningrate = [0.0001,0.0005, 0.001, 0.005, 0.01]
accuracy = [[0.8966666666666666, 0.9026666666666666, 0.9063333333333333, 0.8923333333333333, 0.7073333333333334], [0.897, 0.9013333333333333, 0.904, 0.8963333333333333, 0.897], [0.8986666666666666, 0.902, 0.9066666666666666, 0.9013333333333333, 0.8976666666666666]]
plt.plot(range(len(learningrate)),accuracy[0],'--*')
plt.plot(range(len(learningrate)),accuracy[1],'--*')
plt.plot(range(len(learningrate)),accuracy[2],'--*')
plt.xticks(range(len(learningrate)), learningrate,fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
plt.xlabel('learning rate',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy for 100 Epochs, 3 hidden layers of sizes(500,300,100) with Relu',fontsize=14)
plt.show()



# time = [[15, 5.3, 3.2,2.45,3.3],[14.4,4.8,3.5,2.2,4.1],[13.7,4.1,2.7,1.9,2.8]]
# plt.plot(range(len(learningrate)),time[0],'--*')
# plt.plot(range(len(learningrate)),time[1],'--*')
# plt.plot(range(len(learningrate)),time[2],'--*')
# plt.xticks(range(len(learningrate)), learningrate,fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(['mini batch =100)','mini batch =200','mini batch =500'])
# plt.xlabel('learning rate',fontsize=16)
# plt.ylabel('Run time in minutes',fontsize=16)
# plt.title('Accuracy for neural network with different learning rates after PCA',fontsize=14)
# plt.show()
# plt.title("run time for gradient boosting with different learning rates")
#
# plt.show()