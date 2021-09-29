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
from numpy import *

def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)    # 将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat






#Cluster the users
'''
What criteria should we use to cluster the users?
1. IUIPC or SA-6 scales
2. Demongraphics
3. Acceptability scores

'''



if __name__ == "__main__":
    
    '''
    Get the numerical values for each user data
    '''
    # filename = 'raw_data.csv'
    # all_data = []
    # with open(filename) as f:
    #     reader = csv.reader(f)      
    #     for rows in reader:
    #         row_store = []
    #         for item in rows:
    #             if item == '':
    #                 item = 0
    #             elif item in ['Completely Unacceptable', 'Completely unacceptable']:
    #                 item = 1
    #                 # import pdb; pdb.set_trace()  
    #             elif item in ['Somewhat Unacceptable', 'Somewhat unacceptable']:
    #                 item = 2
    #             elif item in ['Neutral']:
    #                 item = 3
    #             elif item in ['Somewhat Acceptable', 'Somewhat acceptable']:
    #                 item = 4
    #             elif item in ['Completely Acceptable', 'Completely acceptable', 'Completely Acceptable ']:
    #                 item = 5
    #             row_store.append(item)
    #         all_data.append(row_store)
    

    '''
    Write in new file
    '''    
    # file = 'numerical_raw.csv'
    # with open(file, 'w') as f1:
    #     writer = csv.writer(f1)
    #     for rows in all_data:
    #         writer.writerow(rows)

    
    with open('numerical_raw.csv') as fout:
        r = csv.reader(fout)
        l = list(r)
        data = []
        for index, item in enumerate(l):
            if index > 2:
                item = list(map(int, item))
                data.append(item)
          
        k = np.array(data)
        print(k.shape[0]-1)
        import pdb; pdb.set_trace()      
        
        



    




    
        # for index , item in enumerate(rows):
        #     rows[index] = str_to_list(item)
        # all_data.append(rows)
    










