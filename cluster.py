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
    # df = pd.read_csv('./data/sub_data/banking_test.csv', header=None, error_bad_lines=False)
    # df.head()
    
    filename = 'raw_data.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        all_data = []
        for rows in reader:
            import pdb; pdb.set_trace()
            for index , item in enumerate(rows):
                rows[index] = str_to_list(item)
            all_data.append(rows)
        

