import numpy as np
import csv


import pandas as pd
import scipy as sp
import sklearn
from pandas import DataFrame,Series
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn import cluster, datasets
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

with open('means.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    d = DataFrame(data)
    d.head()
print(d)

km = KMeans(n_clusters=2)
km.fit(d)
print(km.cluster_centers_)



# d.to_csv("tzzs_data.csv")
#
# mod = KMeans(n_clusters=2, n_jobs=1, max_iter = 300)
# mod.fit_predict(d)#y_predis the result of clustering
# import pdb; pdb.set_trace()

r1 = pd.Series(km.labels_).value_counts()
r2 = pd.DataFrame(km.cluster_centers_)
r = pd.concat([r2, r1], axis = 1)
r.columns = list(d.columns) + [u'number in the type']
print(r)


r = pd.concat([d, pd.Series(km.labels_, index = d.index)], axis = 1)
r.columns = list(d.columns) + [u'cluster_type']
# print(r.head())
r.to_csv("results.csv")
import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
# with open('lalal.csv', 'w') as fout:
#     writer = csv.writer(fout)
#     writer.writerow(r)
    # for row in r:
        # writer.writerow(r)



visualization
from sklearn.manifold import TSNE
 
ts = TSNE(n_iter=300, verbose=1)
ts.fit_transform(r)
ts = pd.DataFrame(ts.embedding_, index = r.index)

import matplotlib.pyplot as plt
 
a = ts[r[u'cluster_type'] == 0]
plt.plot(a[0], a[1], 'r.')
a = ts[r[u'cluster_type'] == 1]
plt.plot(a[0], a[1], 'go')
a = ts[r[u'cluster_type'] == 2]
plt.plot(a[0], a[1], 'b*')
plt.show()
