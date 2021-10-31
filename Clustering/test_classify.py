



###################################################################
# Classify the test data
#test_1: 124
# test_2: 139
# test_3: 84

# train_1: 621
# train_2: 301
# train_3: 467
###################################################################


import csv
import pandas as pd
import numpy as np

f = open('./data/new_data/num_file_out.csv')
reader = csv.reader(f)
header = next(reader)
true_header = header[1:]
print('Loading data ... ')



type1 = []
type2 = []
type3 = []
# type1.append(header)
# type2.append(header)
# type3.append(header)
for index, value in enumerate(reader):
    if index > 0:      
        if value[0] == '0':
            type1.append(value[1:])         
        elif value[0] == '1':
            type2.append(value[1:])
        elif value[0] == '2':
            type3.append(value[1:])
        else:
            print('Error')
            print(index, value)


print(len(type1))
print(len(type2))
print(len(type3))



################
#Calculate the centroid of clusters
################

data_1 = np.array(type1).astype(float)
data_2 = np.array(type2).astype(float)
data_3 = np.array(type3).astype(float)



print('Calculating centroid array for each cluster ...')
centroid_1 = np.mean(data_1, axis = 0)
centroid_2 = np.mean(data_2, axis = 0)
centroid_3 = np.mean(data_3, axis = 0)

print('Centroid 1 is: ', centroid_1)
print('Centroid 2 is: ', centroid_2)
print('Centroid 3 is: ', centroid_3)



####################
#Load test data
####################

f_test = open('./data/new_data/test_set.csv')
reader = csv.reader(f_test)
test_1 = []
test_2 = []
test_3 = []



for index, item in enumerate(reader):
    dist = []
    if index > 0:
        test_item = np.array(item).astype(float)
        
        dist_1 = np.linalg.norm(test_item - centroid_1)
        dist_2 = np.linalg.norm(test_item - centroid_2)
        dist_3 = np.linalg.norm(test_item - centroid_3)

        dist.append(dist_1)
        dist.append(dist_2)
        dist.append(dist_3)
        # print(dist)
        closest_ = dist.index(min(dist)) + 1
        if closest_ == 1:
            test_1.append(item)
        elif closest_ == 2:
            test_2.append(item)
        elif closest_ == 3:
            test_3.append(item)
        else:
            print('Error')

print('test_1:', len(test_1))
print('test_2:', len(test_2))
print('test_3:', len(test_3))

# import pdb; pdb.set_trace()





###################################
#Save training data without lables
###################################
type1.insert(0,true_header)
type2.insert(0,true_header)
type3.insert(0,true_header)


file1 = './data/new_data/model1_train.csv'
file2 = './data/new_data/model2_train.csv'
file3 = './data/new_data/model3_train.csv'
with open(file1,'w') as fout_1:
        writer = csv.writer(fout_1)
        for rows in type1:
            writer.writerow(rows)

with open(file2,'w') as fout_2:
        writer = csv.writer(fout_2)
        for rows in type2:
            writer.writerow(rows)

with open(file3,'w') as fout_3:
        writer = csv.writer(fout_3)
        for rows in type3:
            writer.writerow(rows)


###################################
#Save test data without lables
###################################


test_1.insert(0,true_header)
test_2.insert(0,true_header)
test_3.insert(0,true_header)


ff1 = './data/new_data/model1_test.csv'
ff2 = './data/new_data/model2_test.csv'
ff3 = './data/new_data/model3_test.csv'


with open(ff1,'w') as fout_11:
        writer = csv.writer(fout_11)
        for rows in test_1:
            writer.writerow(rows)

with open(ff2,'w') as fout_22:
        writer = csv.writer(fout_22)
        for rows in test_2:
            writer.writerow(rows)

with open(ff3,'w') as fout_33:
        writer = csv.writer(fout_33)
        for rows in test_3:
            writer.writerow(rows)

# import pdb; pdb.set_trace()

