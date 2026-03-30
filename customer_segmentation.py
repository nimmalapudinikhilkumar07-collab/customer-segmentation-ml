import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("customer_data.csv")
print(df.head())

df.drop_duplicates(inplace=True)
print(df.isnull().sum())
df.fillna(0)
print(df.describe().T)
print(df["Annual Income (k$)"].max())

# normal visualization for annual income and spending score
plt.scatter(df["Annual Income (k$)"],df["Spending Score (1-100)"])
plt.title("annual income vs spending score")
plt.xlabel("Annual income in dollars k$")
plt.ylabel("spending score of customer")
plt.show()

# now selecting the next step is feature selection

x=df[["Annual Income (k$)","Spending Score (1-100)"]]

scalar = StandardScaler()
x_scaled= scalar.fit_transform(x)

inertia = []

for n in range(1,11):
    kmeans=KMeans(n_clusters=n,tol=0.0001,max_iter=300,random_state=43)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11),inertia,marker="*")
plt.title("elbow method")
plt.xlabel("number of clusters")
plt.ylabel("inertia")
plt.show()

kmeans_5=KMeans(n_clusters=5,max_iter=300,random_state=43)
algo5=kmeans_5.fit_predict(x_scaled)

plt.scatter(x.iloc[:,0],x.iloc[:,1],c=algo5,cmap='rainbow')
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("customer segmentation for k=5")
plt.show()

centers=kmeans_5.cluster_centers_

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=algo5,cmap='rainbow')
plt.scatter(centers[:,0],centers[:,1],color="black",marker='X',s=200,label='centroids')

plt.xlabel("annual income scaled")
plt.ylabel("spending score scaled")
plt.title("customer segmentation for k=5")
plt.legend()
plt.show()

df["cluster"]=algo5

print(df.head())

print(df.groupby("cluster").mean(numeric_only=True))








