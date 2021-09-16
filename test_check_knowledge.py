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


knowledge_base = {'voice recording': {'anonimised': ['primary_user'], 'notified': []}, 
    'shopping': {'anonimised': [], 'notified': [['send', 'spa', 'parents', 'online_shopping', 'shopping', 'primary_user', 'with the purpose of knowing the data']]}, 
    'weather': {'anonimised': [], 'notified': []}}

anonimised_datas = []
notified_datas = []

action = ['send', 'spa', 'other skills', 'to_do', 'voice recording', 'primary_user', '_']
#action = ['send', 'spa', 'parents', 'online_shopping', 'shopping', 'primary_user', 'with the purpose of knowing the data']

key_word_data = action[4]

if key_word_data != '_':
    for item in knowledge_base[key_word_data]['anonimised']:  #[shoud be data, subject]
        if action[5] == item:
            anonimised_datas.append(item)
            import pdb; pdb.set_trace()

    for item in knowledge_base[key_word_data]['notified']:   #should be action
        if item == action[5]:
            notified_datas.append(item)




