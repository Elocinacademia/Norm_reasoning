import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sea
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import csv







#Cluster the users
'''
What criteria should we use to cluster the users?
1. IUIPC or SA-6 scales
2. Demongraphics
3. Acceptability scores

'''



if __name__ == "__main__":
    #Acceptability Scores
    df = pd.read_csv('./data/sub_data/banking_test.csv', header=None, error_bad_lines=False)
    df.head()
    import pdb; pdb.set_trace()
    col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features = pd.DataFrame(features, columns = col_names)
    scaled_features.head()

    with open() as f:
        gender = df['Gender']
        newdf = scaled_features.join(gender)

        newdf = pd.get_dummies(newdf, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)

        newdf = newdf.drop(['Gender_Male'],axis=1)

        newdf.head()

        SSE = []

        for cluster in range(1,10):
            kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
            kmeans.fit(newdf)
            SSE.append(kmeans.inertia_)

        # converting the results into a dataframe and plotting them

            frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
            plt.figure(figsize=(12,6))
            plt.plot(frame['Cluster'], frame['SSE'], marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')



