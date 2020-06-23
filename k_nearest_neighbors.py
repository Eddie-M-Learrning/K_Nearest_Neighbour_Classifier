# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.drop('Purchased' , axis = 'columns')
y = dataset['Purchased']
# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X,y)

# Predicting a new result
print(classifier.predict([[32,150000]]))
print(classifier.score(X,y))


y_pred = classifier.predict(X)
print(y_pred)

ds = pd.read_csv('sample.csv')

s = classifier.predict(ds)
ds['Predicted_Purchases'] = s
print(ds)
#ds.to_csv("samples.csv" , index = False)
