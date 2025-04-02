import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



df=pd.read_csv('C:\ML_folder\K-mean_Cluster.csv')
print(df)

x=df[['X','Y']]
print(x)


wcss=[]
for i in range(1,6):
       Kmean=KMeans(n_clusters=i,init='k-means++',random_state=2)
       Kmean.fit(x)
       wcss.append(Kmean.inertia_)
print(wcss)

x['Cluster Name']=Kmean.fit_predict(x)
print(x)

plt.plot(range(1,6),wcss)
plt.show()
