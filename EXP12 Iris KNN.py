import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv(r"C:\Users\subak\OneDrive - Saveeth School of Engineering\Documents\ITA0634\DATASET\IRIS.csv")

print(iris.head())    
print(iris.describe())

print("Target Labels", iris["species"].unique())

import plotly.io as io
io.renderers.default='browser'

import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

x = iris.drop("species", axis=1)
y = iris["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
"""
from sklearn import metrics
print("Prediction: {}".format(prediction))
print ("-------------------------------------------------------------------------")
print("\nConfusion Matrix:\n",metrics.confusion_matrix(y_test, ypred))  
print ("-------------------------------------------------------------------------")
print("\nClassification Report:\n",metrics.classification_report(y_test, ypred)) 
print ("-------------------------------------------------------------------------")
print('Accuracy of the classifer is %0.2f' % metrics.accuracy_score(y_test,ypred))
print ("-------------------------------------------------------------------------")
"""
