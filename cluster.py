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
    all_data = []
    with open(filename) as f:
        reader = csv.reader(f)      
        for rows in reader:
            row_store = []
            for item in rows:
                if item == '':
                    item = 0
                elif item in ['Completely Unacceptable', 'Completely unacceptable']:
                    item = 1
                    # import pdb; pdb.set_trace()  
                elif item in ['Somewhat Unacceptable', 'Somewhat unacceptable']:
                    item = 2
                elif item in ['Neutral']:
                    item = 3
                elif item in ['Somewhat Acceptable', 'Somewhat acceptable']:
                    item = 4
                elif item in ['Completely Acceptable', 'Completely acceptable', 'Completely Acceptable ']:
                    item = 5
                row_store.append(item)
            all_data.append(row_store)
    

    # import pdb; pdb.set_trace()      
    file = 'numerical_raw.csv'
    with open(file, 'w') as f1:
        writer = csv.writer(f1)
        for rows in all_data:
            writer.writerow(rows)


    import pdb; pdb.set_trace()
        # for index , item in enumerate(rows):
        #     rows[index] = str_to_list(item)
        # all_data.append(rows)
    










