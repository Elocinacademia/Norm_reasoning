
from pprint import pprint
import numpy as np
import pandas as pd



initial_rules = [[[ 'voice recording', 'partner', 'with purpose&condition2'], ['Acceptable'], [1.89, 0.76]], [[ 'voice recording', 'partner', 'with purpose&condition2'], ['Acceptable'], [1.99, 0.99]], [['prime user', 'voice recording', 'with purpose&condition2'], ['Uncceptable'], [1.44, 0.52]], [['prime user', 'voice recording', 'with purpose&condition2'], ['Acceptable'], [1.13, 0.62]], [['call assistant', 'prime user'], ['Acceptable'], [1.17, 0.64]], [['email', 'prime user'], ['Acceptable'], [1.21, 0.66]], [['no purpose&no condition', 'other skills'], ['Acceptable'], [1.25, 0.68]], [['other skills', 'prime user'], ['Acceptable'], [1.25, 0.68]], [['banking', 'prime user'], ['Acceptable'], [1.31, 0.71]], [['no purpose&no condition', 'skills'], ['Unacceptable'], [1.33, 0.61]], [['prime user', 'skills'], ['Unacceptable'], [1.33, 0.61]], [['prime user', 'weather', 'with purpose&condition4'], ['Unacceptable'], [1.39, 0.63]], [['advertising agencies', 'prime user'], ['Acceptable'], [1.44, 0.79]], [['prime user', 'visitors in general'], ['Acceptable'], [1.46, 0.8]], [['voice recording'], ['Acceptable'], [1.13, 0.62]], [['call assistant'], ['Acceptable'], [1.17, 0.64]], [['email'], ['Acceptable'], [1.21, 0.66]], [['other skills'], ['Acceptable'], [1.25, 0.68]], [['banking'], ['Acceptable'], [1.31, 0.71]], [['skills'], ['Unacceptable'], [1.33, 0.61]], [['weather'], ['Unacceptable'], [1.39, 0.77]], [['advertising agencies'], ['Acceptable'], [1.44, 0.79]], [['visitors in general'], ['Acceptable'], [1.46, 0.8]], [['no purpose&no condition', 'other skills', 'prime user'], ['Acceptable'], [1.25, 0.68]], [['no purpose&no condition', 'prime user', 'skills'], ['Unacceptable'], [1.33, 0.61]], [['parents', 'shopping', 'with purpose&condition1'], ['Acceptable'], [1.45, 0.98]] ,[['shopping', 'with purpose&condition5'], ['Unacceptable'], [1.33, 0.38]] ,[['shopping'], ['Unacceptable'], [1.32, 0.57]]]









def takeThird_con(elem):
    '''
    Take second element for sort
    '''
    return elem[2][1]

def takeThird_lif(elem):
    '''
    Take second element for sort
    '''
    return elem[2][0]



actions = [['send', 'spa', 'skills', '_', '_', 'primary_user', 'with the purpose of knowing the data', 4], ['send', 'spa', 'advertising agencies', '_', '_', 'primary_user', 'with the purpose of knowing the data', 4], ['send', 'spa', 'parents', '_', '_', 'primary_user', '_', 4], ['send', 'spa', 'advertising agencies', '_', '_', 'primary_user', 'with the purpose of knowing the data', 4]]
matching_norm_base = {('send', 'spa', 'skills', '_', '_', 'primary_user', '_'): {'F': [[1.16, 0.63]]}, ('send', 'spa', 'skills', '_', '_', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.45, 0.79]]}, ('send', 'spa', 'advertising agencies', '_', '_', 'primary_user', 'with the purpose of knowing the data'): {'F': [[1.49, 0.81]]}, ('send', 'spa', 'advertising agencies', '_', '_', 'primary_user', '_'): {'F': [[1.66, 0.9], [1.47, 0.8]]}, ('send', 'spa', 'house keeper', '_', '_', 'primary_user', '_'): {'F': [[1.07, 0.58]]}, ('send', 'spa', 'close family', '_', '_', 'primary_user', '_'): {'P': [[1.23, 0.56]]}, ('send', 'spa', 'parents', '_', '_', 'primary_user', '_'): {'P': [[1.33, 0.61]]}, ('send', 'spa', 'children', '_', '_', 'primary_user', '_'): {'P': [[1.41, 0.64]]}, ('send', 'spa', 'neighbours', '_', '_', 'primary_user', '_'): {'F': [[1.46, 0.79]]}, ('send', 'spa', 'visitors in general', '_', '_', 'primary_user', '_'): {'F': [[1.47, 0.79]]}, ('send', 'spa', 'partner', '_', '_', 'primary_user', '_'): {'P': [[1.68, 0.77]]}}


decision_list = []
most_similar_one = []

for index, act in enumerate(actions):
    tempo = []
    real_act = act[:-1]
    tempo.append(real_act)
    for key, value in matching_norm_base[tuple(real_act)].items():
        tempo.append(key)
        tempo.append(value[0])
    decision_list.append(tempo)
decision_list.sort(key=takeThird_con, reverse=True)
x = decision_list[0][2][1]  #compare the confidence value
to_judge_lift = []


for value in decision_list:
    if value[2][1] == x:
        to_judge_lift.append(value)

if len(to_judge_lift) == 1:
    most_similar_one = to_judge_lift[0][0]
else:
    to_judge_lift.sort(key=takeThird_lif, reverse=True)
    y = to_judge_lift[0][2][0]  #compare the lift value
    final_round = []
    for value1 in to_judge_lift:
        if value1[2][0] == y:
            final_round.append(value1)
    if len(final_round) == 1:
        most_similar_one = final_round[0][0]
    else:
        '''
        if more than one norms have same lift and confidence value
        Compare the number of P and F
        if n(P) > F:
            return the first action with P
        else:
            return the firsr action with F
        '''
        permission = 0
        prohibition = 0
        allow_list_index = []
        reject_list_index = []
        for index ,item in enumerate(final_round):
            if item[1] == 'p':
                permission += 1
                allow_list_index.append(index)
            if item[1] == 'F':
                prohibition +=1
                reject_list_index.append(index)
        if prohibition >= permission:
            f1 = reject_list_index[0]
            most_similar_one = final_round[f1][0]
        else:
            f2 = allow_list_index[0]
            most_similar_one = final_round[f2][0]


y = tuple(most_similar_one) 
import pdb; pdb.set_trace()