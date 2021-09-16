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



norm_base = {'email': {'_': {'_': {('send', 'spa', '_', 'email', 'email', 'primary_user', '_'): {'F': [[1.21, 0.66]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'smart camera': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'to do list': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'sleeping hours': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'banking': {'_': {'_': {('send', 'spa', '_', 'banking', 'banking', 'primary_user', '_'): {'F': [[1.31, 0.71]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'healthcare': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'door locker': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'call assistant': {'_': {'_': {('send', 'spa', '_', 'call_assistant', 'call assistant', 'primary_user', '_'): {'F': [[1.17, 0.64]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'video call': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'location': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'voice recording': {'_': {'_': {('send', 'spa', '_', 'voice_recordings', 'voice recording', 'primary_user', '_'): {'F': [[1.13, 0.62]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {'primary_user': {('send', 'spa', 'partner', 'voice_recordings', 'voice recording', 'primary_user', 'with the purpose of knowing the data'): {'F': [1.99, 0.99]}, ('send', 'spa', '_', 'voice_recordings', 'voice recording', 'primary_user', 'with the purpose of knowing the data'): {'F': [1.13, 0.62]}}}, 'notified': {}}, 'playlists': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'thermostat': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'shopping': {'_': {'_': {('send', 'spa', '_', 'online_shopping', 'shopping', 'primary_user', '_'): {'F': [[1.32, 0.57]]}}, 'confidential': {}, 'review': {('send', 'spa', '_', 'online_shopping', 'shopping', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.33, 0.38]]}}, 'store': {}}, 'anonimised': {}, 'notified': {'primary_user': {('send', 'spa', 'parents', 'online_shopping', 'shopping', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.45, 0.98]]}}}}, 'weather': {'_': {'_': {('send', 'spa', '_', 'weather_forecast', 'weather', 'primary_user', '_'): {'F': [[1.39, 0.77]]}}, 'confidential': {}, 'review': {}, 'store': {('send', 'spa', '_', 'weather_forecast', 'weather', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.39, 0.63]]}}}, 'anonimised': {}, 'notified': {}}, '_': {'_': {'_': {('send', 'spa', 'other skills', '_', '_', 'primary_user', '_'): {'F': [[1.25, 0.68]]}, ('send', 'spa', 'skills', '_', '_', 'primary_user', '_'): {'F': [[1.33, 0.61]]}, ('send', 'spa', 'advertising agencies', '_', '_', 'primary_user', '_'): {'F': [[1.44, 0.79]]}, ('send', 'spa', 'visitors in general', '_', '_', 'primary_user', '_'): {'F': [[1.46, 0.8]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}}


knowledge_base = {'voice recording': {'anonimised': ['primary_user', 'alice'], 'notified': []}, 
    'shopping': {'anonimised': [], 'notified': [['send', 'spa', 'parents', 'online_shopping', 'shopping', 'primary_user', 'with the purpose of knowing the data']]}, 
    'weather': {'anonimised': [], 'notified': []}}

anonimised_datas = []
notified_datas = []

action = ['send', 'spa', 'other skills', 'to_do', 'voice recording', 'primary_user', '_']
# action = ['send', 'spa', 'parents', 'online_shopping', 'shopping', 'primary_user', 'with the purpose of knowing the data']

key_word_data = action[4]

if key_word_data != '_':
    for item in knowledge_base[key_word_data]['anonimised']:  #[shoud be data, subject]
        if action[5] == item:
            anonimised_datas.append(item)
    for item in knowledge_base[key_word_data]['notified']:   #should be action
        if item == action:
            notified_datas.append('true')  #don't need to append the action again, we only need to inform that this precondition was satisfied

active_norm_base = {}
empty = []

if anonimised_datas != empty:
    for item in anonimised_datas:
        if norm_base[key_word_data]['anonimised'][item]:
            for keys, values in norm_base[key_word_data]['anonimised'][item].items():
                active_norm_base[keys] = values
                import pdb; pdb.set_trace()
if notified_datas[0] == 'true':
    for key1, value1 in norm_base[key_word_data]['notified'].items():
        for key2, value2 in value1.items():
            if list(key2) == action:
                active_norm_base[key2] = value2
import pdb; pdb.set_trace()

        

#     for item in notified_datas:
#         if norm_base[key_word_data]['notified'][item]:
#             for keys, values in norm_base[key_word_data]['notified'][item].items():
#                 active_norm_base[keys] = values
# if norm_base[key_word_data]['_']['_'] != {}:
#     for keys, values in norm_base[key_word_data]['_']['_'].items():
#         active_norm_base[keys] = values





