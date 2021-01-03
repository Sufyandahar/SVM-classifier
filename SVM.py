# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dataset = pd.read_csv('Iris.csv')
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
dataset.head()

# slicing the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# spliting the dataset into traning and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)



# SVM Classifer
from sklearn.svm import SVC
classifier = SVC(kernel='linear')

# fit the model
classifier.fit(X_train, y_train)

# prediction on test data
y_pred = classifier.predict(X_test)

# check accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_score :", accuracy_score(y_pred, y_test))

##########################################################  Prediction ##############################################################

result = classifier.predict([[4.9,3.0,1.4,0.2]])
print(result)

