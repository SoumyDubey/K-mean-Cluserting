#!/usr/bin/env python
# coding: utf-8

# # Prediction Using Unsupevised Machine Learning

# Task: From the given "Iris" dataset, predict the optimum number of clusters and represt in visually.

# Python used to perform task.

# Task completed during Data Science & Analytics Internship @ The Sparks Foundation

# By- Soumy Dubey

# Links:

# In[1]:


# Import Libraries

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import warnings


# ## Loading the Iris dataset

# In[2]:


dataset = datasets.load_iris()


# In[3]:


iris_df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
iris_df.head(10)


# # Finding the Optimal Number of clusters.

# In[4]:


# Finding the optimal number of cluster for k-mean classifiction.

X = iris_df.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# # Plotting the graph into a line graph to observe the pattern

# In[5]:


# Plotting the graph onto a line graph allow us to abserve 'The Elbow'

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of sqaures
plt.show()


# # Creating K-Mean Classifier

# In[7]:


# Applying kmeans to the dataset 
# Creating the kmeans classifier

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# # Vasualizing the cluster data

# In[9]:


# Visualising the clusters 
# Preferably on the first two columns
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')


# In[10]:


# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# # Combining the graphs

# In[12]:


# Visualising the clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




