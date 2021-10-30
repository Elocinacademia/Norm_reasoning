import sys, os
import copy as c

import numpy as np
import hdbscan
from tsne_python.tsne import tsne
from sklearn import cluster, preprocessing, metrics
from scipy import ndimage
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import random
from random import choice


random.seed(2)

# create affinity matrix
def compute_affinity_matrix(embeddings, dis_type='cos'):
    if dis_type == 'cos':
        # This is the: cosine similarity = 1 - cosine distance
        affinity_matrix = (1 + metrics.pairwise.cosine_similarity(embeddings))/2;
        # for i in range(affinity_matrix.shape[0]):
        #     affinity_matrix[i,i] = 0;
        #     affinity_matrix[i,i] = np.max(affinity_matrix[i])
    elif dis_type == 'euc':
        affinity_matrix = metrics.pairwise.euclidean_distances(embeddings)
    else:
        raise Exception('Unrecognised distance type')
    return affinity_matrix;

def compute_num_clusters(eigenvalues, min_cluster_num = 2, sort = True):
    copy_eigenvalues = c.copy(eigenvalues)

    if sort:
        copy_eigenvalues.sort()
        copy_eigenvalues = copy_eigenvalues[::-1]

    # Remove first n eigenvalues:
    remove_eigenvals = min_cluster_num - 1
    copy_eigenvalues = copy_eigenvalues[remove_eigenvals:]

    # Get ratio
    eigenvalue_ratio = copy_eigenvalues[:-1]/copy_eigenvalues[1:]

    num_clusters = int(np.argmax(eigenvalue_ratio)) + remove_eigenvals + 1
    #return int(max(3, num_clusters))
    return num_clusters

numbers = []
print('Loading data...')








with open('./data/new_data/training_set.csv') as fin:
    for i, line in enumerate(fin):
        if i == 0:
            headers = [item for item in line.split(',')]
        else:
            numbers.append([int(value) for value in line.split(',')])

data = np.array(numbers)
print('Data loaded:')
print(data)

# print('Performing HDBscan clustering...')
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
# cluster_labels = clusterer.fit_predict(data)

#####################################################################
# Start Clustering
#####################################################################
print('Perform spectral clustering..')
min_cluster_num, max_cluster_num = 2, 7
# affinity_matrix = compute_affinity_matrix(data, dis_type='cos')
affinity_matrix = compute_affinity_matrix(data, dis_type='euc')
eigenvalues = eigs(affinity_matrix, k = max_cluster_num + 1, return_eigenvectors = False)
num_clusters = compute_num_clusters(np.real(eigenvalues), min_cluster_num = min_cluster_num)
print("Eigenvalues: ")
print(np.real(eigenvalues))
print("Number of clusters = {}".format(num_clusters))

# Creating the cluster algorithm:
spectral = cluster.SpectralClustering(n_clusters = num_clusters, eigen_solver = 'arpack',
                                      n_init = 10, affinity = 'precomputed', eigen_tol = 5.0)
# Train it on the processed affinity matrix:
spectral.fit(affinity_matrix)


# import pdb; pdb.set_trace()


cluster_labels = spectral.labels_.astype(np.int)

#####################################################################
# Data Visualisation
#####################################################################
# print(max(cluster_labels))
# print('t-SNE plot')
# Y = tsne(data, no_dims=2, initial_dims=5, perplexity=30.0)
# plt.scatter(Y[:, 0], Y[:, 1], c=cluster_labels)
# # plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)
# plt.show()

#####################################################################
# Data Write Out
#####################################################################
with open('./data/new_data/num_file_out.csv', 'w') as fout:
    headers.insert(0,'label')
    fout.write(','.join(headers) + '\n')
    for i, nums in enumerate(numbers):
        fout.write('{},'.format(cluster_labels[i])+','.join([str(n) for n in nums]) + '\n')
