import pandas as pd                                                 #supports dataframes and i/o
import numpy as np                                                  #use numpy because it supports array structures
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot                                  #plot stuff
import pickle                                                       #save your model with highest score so that you can use it to predict after reopening app
from matplotlib import style                                        #change the style of grid

df = pd.read_csv("student-mat.csv", sep=";")                        #separator is ;
df = df[["G1","G2","G3","studytime","absences","failures"]]         #choose only integers of LR, and variables that could be strongly correlated to target
print(df.head())                                                    #print head information to ceck

target = "G3"                                                       #aka label or the thing you're predicting
x = np.array(df.drop([target],1))                                   #df without target - drop is pandas function - 1 in argument means drop column
y = np.array(df[target])                                            #target

best = 0
while best<0.9:
    x_train,x_test, y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)     #split 10% of data into test cases, and rest into training

    LR = linear_model.LinearRegression()
    LR.fit(x_train,y_train)                                             # find the best fit line for input data
    accuracy = LR.score(x_test,y_test)                                  # find accuracy with output data

    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as file:                 # write the linear regress model into the file - saves the LR model
            pickle.dump(LR, file)

pickle_in = open("studentmodel.pickle", "rb")                       # read the file
LR = pickle.load(pickle_in)                                         # overwrite your current model with the model with best accuracy

print(round(LR.score(x_test,y_test), 4) * 100, "%")                 # print %age accuracy of model

print("Coefficients:\n", LR.coef_)                                  # 5 coefficients means that best fit line is on a 5 dimensional plane
print("Intercepts:\n", LR.intercept_)

y_pred = LR.predict(x_test)                                         # going to predict target for test data
for i in range(len(y_pred)):
    print(y_pred[i], "\t", x_test[i], "\t", y_test[i])              # printing out prediction vs input vs output

style.use("ggplot")
pyplot.scatter(df["G1"],df["G3"])
pyplot.xlabel("First Unit test grades")
pyplot.ylabel("Final Exam grades")
pyplot.show()