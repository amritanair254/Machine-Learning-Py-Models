import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("car.data")
print(df.head())


#PREPROCESSING to convert all categorical variables to numeric integrs for knn to process
enc = preprocessing.LabelEncoder()
buying = enc.fit_transform(list(df["buying"]))
maint = enc.fit_transform(list(df["maint"]))
door = enc.fit_transform(list(df["door"]))
persons = enc.fit_transform(list(df["persons"]))
lug_boot = enc.fit_transform(list(df["lug_boot"]))
safety = enc.fit_transform(list(df["safety"]))
cls = enc.fit_transform(list(df["class"]))

x = list(zip(buying,maint,door,persons,lug_boot,safety))                # zip creates a tuple of all the input lists
y = cls

#MODEL
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

KNN = KNeighborsClassifier(n_neighbors=5)                               # for this dataset, k= 9 gives more accuracy
KNN.fit(x_train, y_train)
accuracy = KNN.score(x_test, y_test)
print(str(round(accuracy*100,2)),"%")

#PREDICT
names = ["unacc","acc","good","vgood"]
y_pred = KNN.predict(x_test)
for i in range(len(y_pred)):
    print(x_test[i], "\t", names[y_pred[i]] , "\t", names[y_test[i]] )
    # print(KNN.kneighbors([x_test[i]], 5 , True) )
    # returns indices of and distances from the neighbors for each point.
    # 1st param is 2d list of points, 2nd is k, 3rd is whether to return the distance from the neighbors in addition to the indices of the neighbors

#METRICS
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

#CHOOSING OPTIMAL K
error_rate = []
for i in range (1,15):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(x_train, y_train)
    y_pred = KNN.predict(x_test)
    error_rate.append(np.mean(y_pred != y_test))

#PLOT THE ELBOW GRAPH
plt.figure(figsize =(10,6))
plt.plot(range(1,15), error_rate)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()                                      #Optimal k is 7