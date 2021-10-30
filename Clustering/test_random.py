
import copy
import random
import csv
from random import choice

##############################################################
#  The data first being seperated in 8:2 ratio.
#  8 is the training set, 2 is the test set.
##############################################################

filename = './data/new_data/num_file.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header = next(reader)
    all_data = []
    for rows in reader:
        all_data.append(rows)
    
random.seed(1)

random.shuffle(all_data)
cut = round(len(all_data)*0.8)

training_set = []
test_set = []
training_set = all_data[:cut-1]
test_set = all_data[cut:]


training_set.insert(0, header)
test_set.insert(0, header)

print('training set:', len(training_set))
print('test set:', len(test_set))

# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
file1 = './data/new_data/training_set.csv'
file2 = './data/new_data/test_set.csv'

with open(file1, 'w') as fout:
    writer = csv.writer(fout)
    # for rows in header:
    #     writer.writerow(rows)
    for row in training_set:
        writer.writerow(row)

with open(file2, 'w') as ff:
    writer = csv.writer(ff)
    # for rows in header:
    #     writer.writerow(rows)
    for row in test_set:
        writer.writerow(row)




