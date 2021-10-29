import hdbscan
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data, _ = make_blobs(1000)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data)
print(data)
x = data[:, 0]
y = data[:, 1]
import pdb; pdb.set_trace()
plt.scatter(x, y, cluster_labels)
plt.show()
