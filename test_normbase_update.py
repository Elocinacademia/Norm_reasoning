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


norm_base = {'email': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'smart camera': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'to do list': {'_': {'_': {('send', 'spa', '_', 'to_do', 'to do list', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.12, 0.61]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'sleeping hours': {'_': {'_': {('send', 'spa', '_', 'sleep_aid', 'sleeping hours', 'primary_user', '_'): {'F': [[1.09, 0.59]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'banking': {'_': {'_': {('send', 'spa', '_', 'banking', 'banking', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.46, 0.79]]}, ('send', 'spa', '_', 'banking', 'banking', 'primary_user', '_'): {'F': [[1.48, 0.8], [1.31, 0.71]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'healthcare': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'door locker': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'call assistant': {'_': {'_': {('send', 'spa', '_', 'call_assistant', 'call assistant', 'primary_user', '_'): {'F': [[1.18, 0.64]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'video call': {'_': {'_': {('send', 'spa', '_', 'video_calls', 'video call', 'primary_user', '_'): {'F': [[1.06, 0.58]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'location': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'voice recording': {'_': {'_': {('send', 'spa', '_', 'voice_recordings', 'voice recording', 'primary_user', '_'): {'F': [[1.14, 0.62]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'playlists': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'thermostat': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'shopping': {'_': {'_': {}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, 'weather': {'_': {'_': {('send', 'spa', '_', 'weather_forecast', 'weather', 'primary_user', '_'): {'P': [[1.37, 0.63]]}}, 'confidential': {}, 'review': {}, 'store': {}}, 'anonimised': {}, 'notified': {}}, '_': {'_': {'_': {('send', 'spa', 'skills', '_', '_', 'primary_user', '_'): {'F': [[1.16, 0.63]]}, ('send', 'spa', 'skills', '_', '_', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.45, 0.79]]}, ('send', 'spa', 'advertising agencies', '_', '_', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.64, 0.89]]}, ('send', 'spa', 'advertising agencies', '_', '_', 'primary_user', '_'): {'F': [[1.66, 0.9], [1.47, 0.8]]}, ('send', 'spa', 'house keeper', '_', '_', 'primary_user', '_'): {'F': [[1.07, 0.58]]}, ('send', 'spa', 'close family', '_', '_', 'primary_user', '_'): {'P': [[1.23, 0.56]]}, ('send', 'spa', 'parents', '_', '_', 'primary_user', '_'): {'P': [[1.33, 0.61]]}, ('send', 'spa', 'children', '_', '_', 'primary_user', '_'): {'P': [[1.41, 0.64]]}, ('send', 'spa', 'neighbours', '_', '_', 'primary_user', '_'): {'F': [[1.46, 0.79]]}, ('send', 'spa', 'visitors in general', '_', '_', 'primary_user', '_'): {'F': [[1.47, 0.79]]}, ('send', 'spa', 'partner', '_', '_', 'primary_user', '_'): {'P': [[1.68, 0.77]]}}, 'confidential': {}, 'review': {}, 'store': {('send', 'spa', 'advertising agencies', '_', '_', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.49, 0.81]]}}}, 'anonimised': {}, 'notified': {}}}

def takeSecond(elem):
    '''
    Take second element for sort
    '''
    return elem[1]



for key1, value1 in norm_base.items():
    for key2, value2 in value1.items():
        for key3, value3 in value2.items():
            for key4, value4 in value3.items():
                for key5, value5 in value4.items():
                    if len(value5)>1 :
                        # import pdb; pdb.set_trace()
                        value5.sort(key=takeSecond, reverse=True)
                        new_list = []
                        new_list.append(value5[0])
                        norm_base[key1][key2][key3][key4][key5] = new_list

import pdb; pdb.set_trace()