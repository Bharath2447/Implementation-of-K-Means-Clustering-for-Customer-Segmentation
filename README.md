# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data: Read the dataset using Pandas and explore it with .head() and .info() to understand the structure.
2. Find Optimal Clusters (Elbow Method): Use the Elbow method by fitting KMeans for k values from 1 to 10 and plot the Within-Cluster Sum of Squares (WCSS) to find the optimal number of clusters.

3. Train KMeans Model: Initialize and fit the KMeans model using the chosen number of clusters (e.g., 5) on the selected features (Annual Income and Spending Score).

4. Predict Clusters: Use the trained model to predict cluster labels and add them as a new column to the dataset.

5. Segment Data: Split the data into separate DataFrames based on predicted cluster labels for visualization.

6. Visualize Clusters: Plot the clusters using different colors to visualize how customers are grouped based on income and spending score.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Bharath K
RegisterNumber: 212224230036

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Mall_Customers.csv")
df
df.head()
df.tail()
df.info()
df.isnull().sum()
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init= "k-means++")
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
kn=KMeans(n_clusters=5)
kn.fit(df.iloc[:,3:])
y_pred= kn.predict(df.iloc[:,3:])
y_pred
df["cluster"]=y_pred
df0=df[df["cluster"]==0]
df1=df[df["cluster"]==1]
df2=df[df["cluster"]==2]
df3=df[df["cluster"]==3]
df4=df[df["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
  
*/
```

## Output:

df:

<img width="545" height="383" alt="image" src="https://github.com/user-attachments/assets/760c0378-813c-44ec-bd8a-c0f93a670de8" />

df.head():

<img width="532" height="180" alt="image" src="https://github.com/user-attachments/assets/351b8026-6198-4933-b8be-4e6fcc61bdb6" />

df.tail():

<img width="562" height="185" alt="image" src="https://github.com/user-attachments/assets/6693c143-1828-418c-83d5-adad33d3d57a" />

df.info():

<img width="476" height="247" alt="image" src="https://github.com/user-attachments/assets/3cca634f-7995-48fd-a3f8-f7751a52ade9" />

df.isnull().sum():

<img width="257" height="127" alt="image" src="https://github.com/user-attachments/assets/c49d00fd-d986-4bdb-ad00-abc3e0b7e829" />

graph:

<img width="755" height="553" alt="image" src="https://github.com/user-attachments/assets/1b9e4952-714b-4af4-8991-f0527dcf10f5" />

kn.fit(df.iloc[:,3:]):

<img width="786" height="80" alt="image" src="https://github.com/user-attachments/assets/3e9061ce-2675-4481-a8ed-487841b79f1e" />

y_pred:

<img width="665" height="205" alt="image" src="https://github.com/user-attachments/assets/ff41d35c-838f-4a94-b4ed-d9beed83bbbc" />

Customer Segments:

<img width="643" height="536" alt="image" src="https://github.com/user-attachments/assets/265fa549-7108-4169-835a-552b061b9dd0" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
