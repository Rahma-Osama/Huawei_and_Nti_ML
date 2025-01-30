import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
#we'll test on only 2 columns to ease the visualization process
x=dataset.iloc[:,3:5] 

from sklearn.cluster import KMeans
wcss=[]

#we need to run kmeans several times to choose the best num of clusters
#we slect the second curve from the elbow method
for i in range(1,21):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,21),wcss)    
plt.title('The Elbow Method')
plt.xlabel('Num of clusters')
plt.ylabel('WCSS')
plt.show()

# from the figure we'll choose 5 clusters

#now apply the algo and predct y
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)

#Now visualization Time!!
plt.scatter(x.iloc[y_kmeans == 0, 0], x.iloc[y_kmeans == 0, 1], s = 50, c = 'red', label='Cluster 1')
plt.scatter(x.iloc[y_kmeans == 1, 0], x.iloc[y_kmeans == 1, 1], s = 50, c = 'blue', label='Cluster 2')
plt.scatter(x.iloc[y_kmeans == 2, 0], x.iloc[y_kmeans == 2, 1], s = 50, c = 'green', label='Cluster 3')
plt.scatter(x.iloc[y_kmeans == 3, 0], x.iloc[y_kmeans == 3, 1], s = 50, c = 'orange', label='Cluster 4')
plt.scatter(x.iloc[y_kmeans == 4, 0], x.iloc[y_kmeans == 4, 1], s = 50, c = 'cyan', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 50, c = 'yellow')
plt.show()
