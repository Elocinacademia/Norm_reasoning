import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import re
import pandas as pd
import random
import copy
import math
from random import choice

def takeSeventh(elem):
    '''
    Take sixth element for sort
    '''
    return elem[7]




def parameter_calculation(action1, action2):
    count = 0
    if len(action1) == len(action2):
        for index, element in enumerate(action1):
            if element == action2[index]:
                count +=1
    else:
        print("Wrongly formatted action found!", action1)
    
    return count


def find_most_similar_action_2(actions, this_action):
    '''
    This is the first type of method to find the most similar action.
    *** How many parameters they have are same ***
    如果最大的值有多个：action1 和action2都有5个parameters和this_action相同怎么办 所以应该返回一个list

    '''
    collections = []
    for value in actions:
        l = []
        same_parameters = parameter_calculation(value, this_action)
        l = copy.deepcopy(value)
        l.append(same_parameters)
        collections.append(l)

    collections.sort(key=takeSeventh, reverse=True)
    # import pdb;pdb.set_trace()
    x = collections[0][-1]
    most_similar = []
    for value in collections:
        if value[-1] == x:
            most_similar.append(value)

    return most_similar, len(most_similar)




new_action = ['send', 'spa', 'other skills', 'to_do', 'to do list', 'primary_user', '_']
possible_actions = [['send', 'spa', '_', 'to_do', 'to do list', 'primary_user', 'with the purpose of knowing the data'], ['send', 'spa', 'skills', '_', '_', 'primary_user', '_'], ['send', 'spa', 'skills', '_', '_', 'primary_user', 'with the purpose of knowing the data'], ['send', 'spa', '_', 'banking', 'banking', 'primary_user', 'with the purpose of knowing the data'], ['send', 'spa', '_', 'banking', 'banking', 'primary_user', '_']]


(x, y) = find_most_similar_action_2(possible_actions, new_action)
print(x)
import pdb;pdb.set_trace()







