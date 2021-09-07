
import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import re
import pandas as pd
import random
import copy



from efficient_apriori import apriori

random.seed(1)

initial_rules = [[[ 'voice recording', 'partner', 'with purpose&condition2'], ['Acceptable'], [1.99, 0.99]], [['prime user', 'voice recording', 'with purpose&condition2'], ['Uncceptable'], [1.44, 0.52]], [['prime user', 'voice recording', 'with purpose&condition2'], ['Acceptable'], [1.13, 0.62]], [['call assistant', 'prime user'], ['Acceptable'], [1.17, 0.64]], [['email', 'prime user'], ['Acceptable'], [1.21, 0.66]], [['no purpose&no condition', 'other skills'], ['Acceptable'], [1.25, 0.68]], [['other skills', 'prime user'], ['Acceptable'], [1.25, 0.68]], [['banking', 'prime user'], ['Acceptable'], [1.31, 0.71]], [['no purpose&no condition', 'skills'], ['Unacceptable'], [1.33, 0.61]], [['prime user', 'skills'], ['Unacceptable'], [1.33, 0.61]], [['prime user', 'weather', 'with purpose&condition4'], ['Unacceptable'], [1.39, 0.63]], [['advertising agencies', 'prime user'], ['Acceptable'], [1.44, 0.79]], [['prime user', 'visitors in general'], ['Acceptable'], [1.46, 0.8]], [['voice recording'], ['Acceptable'], [1.13, 0.62]], [['call assistant'], ['Acceptable'], [1.17, 0.64]], [['email'], ['Acceptable'], [1.21, 0.66]], [['other skills'], ['Acceptable'], [1.25, 0.68]], [['banking'], ['Acceptable'], [1.31, 0.71]], [['skills'], ['Unacceptable'], [1.33, 0.61]], [['weather'], ['Unacceptable'], [1.39, 0.77]], [['advertising agencies'], ['Acceptable'], [1.44, 0.79]], [['visitors in general'], ['Acceptable'], [1.46, 0.8]], [['no purpose&no condition', 'other skills', 'prime user'], ['Acceptable'], [1.25, 0.68]], [['no purpose&no condition', 'prime user', 'skills'], ['Unacceptable'], [1.33, 0.61]], [['parents', 'shopping', 'with purpose&condition1'], ['Acceptable'], [1.45, 0.98]] ,[['shopping', 'with purpose&condition5'], ['Unacceptable'], [1.33, 0.38]] ,[['shopping'], ['Unacceptable'], [1.32, 0.57]]]



attributes = {
'smart_camera': ['smart camera'],
'healthcare': ['healthcare'],
'email': ['email'],
'smart_door_locker': ['door locker'],
'banking': ['banking'],
'video_calls': ['video call'],
'voice_recordings': ['voice recording'],
'call_assistant': ['call assistant'],
'ride_service': ['location'],
'to_do': ['to do list'],
'smart_thermostat': ['thermostat'],
'online_shopping': ['shopping'],
'sleep_aid': ['sleeping hours'],
'playlists': ['playlists'],
'weather_forecast': ['weather']
}



data_use_to_create_base = {'data': ['email', 'banking', 'healthcare', 'door locker', 'smart camera', 'call assistant', 'video call', 'location', 'voice recording', 'to do list', 'sleeping hours', 'playlists', 'thermostat', 'shopping', 'weather', '_']}



data_clean = {'camera': 'smart camera',
'todo': 'to do list',
'sleep hours': 'sleeping hours'}


action_dictionary = {
'target': ['parents', 'partner', 'siblings', 'housemates', 'children', 'neighbours', 'close friends', 'close family', 'house keeper', 'visitors in general', 'assistant provider', 'skills', 'other skills', 'advertising agencies', 'law enforcement agencies'], 
'data': ['email', 'banking', 'healthcare', 'door locker', 'smart camera', 'camera', 'todo', 'call assistant', 'video call', 'location', 'voice recording', 'to do list', 'sleeping hours', 'sleep hours', 'playlists', 'thermostat', 'shopping', 'weather'],
'principle': ['no purpose&no condition', 'with purpose&no condition', 'with purpose&condition1', 'with purpose&condition2', 'with purpose&condition3', 'with purpose&condition4', 'with purpose&condition5']
}
# actions = ['name', 'actor', 'target', 'attribute', 'data', 'subject', 'purpose']

trans_recipient = {'your partner': 'partner',
'your siblings': 'siblings',
'your parents': 'parents',
'your housemates': 'housemates',
'your children': 'children',
}





def accuracy_verify(x,y):
    accept = {'F': 'unacceptable',
    'P': 'acceptable'}
    result = accept[x]
    if result == y:
        return '1'
    else:
        return '0'





def get_key_from_dictionary(dict, value):
    return [key for key, v in dict.items() if value in v]



def action_format(rule_left_list):
    '''
    Return a well-formatted action and it's condition type 
    Used to form a relevant norm
    '''
    # rule_left_list[i] = ['call assistant', 'prime user']
    #action[i] should be = ['name', 'actor = 'SPA'', 'target', 'attribute', 'data', 'subject', 'purpose']
    action = ['send', 'spa', '_', '_', '_', 'primary_user', '_']
    condition_type = '_'

    

    for item in rule_left_list:
        if item in trans_recipient.keys():
            item = trans_recipient[item]
        if item in action_dictionary['target']:
            action[2] = item
        if item in action_dictionary['data']:
            if item in data_clean.keys():
                action[4] = data_clean[item]
            else:
                action[4] = item
            my_key = get_key_from_dictionary(attributes, action[4])
            action[3] = my_key[0]
        if item in action_dictionary['principle']:
            '''
            condition1: notified(Actor,Subject,Action)
            condition2: anonimised(Data)
            condition3: confidential〈F,,send(Actor,Target,Attribute,Data,Subject,Purpose)
            condition4: after purpose detele〈O,achieved(Purpose),delete(Actor,Data),〉
            condition5: can review(Actor,Data)
            '''
            if item == 'no purpose&no condition':
                action[6] = '_'
                condition_type = '_'
            elif item == 'with purpose&no condition':
                action[6] = 'with the purpose of knowing the data'
                condition_type = '_'
            elif item == 'with purpose&condition1':
                action[6] = 'with the purpose of knowing the data'
                condition_type = 'notified'
            elif item == 'with purpose&condition2':
                action[6] = 'with the purpose of knowing the data'
                condition_type = 'anonimised'
            elif item == 'with purpose&condition3':
                action[6] = 'with the purpose of knowing the data'
                condition_type = 'confidential'
            elif item == 'with purpose&condition4':
                action[6] = 'with the purpose of knowing the data'
                condition_type ='store'
            elif item == 'with purpose&condition5':
                action[6] = 'with the purpose of knowing the data'
                condition_type = 'review'

    return action, condition_type
    
  


class UserFeedback:
    '''
    this is used to form the users' feedback
    '''
    def __init__(self, body):
        pass





def data_generator(filename):
    """
    Data generator, needs to return a generator to be called several times.
    Use this approach if data is too large to fit in memory. If not use a list.
    """
    def data_gen():
        with open(filename) as file:
            for line in file:
                yield tuple(k.strip() for k in line.split(','))
    
    return data_gen

def rule_mining(min_support, min_confidence, target_datatype):
    '''
    Together with data_generator, used to mind the association rules
    Return a set of rules that agreed by the majority of people, together with decisions made on them and [lift+confidence] value
    '''
    file1 = './data/sub_data/' + target_datatype + '_train.csv'
    transactions = data_generator(file1)
    itemsets, rules = apriori(transactions, min_support, min_confidence)

    # Print out every rule with 2 items on the left hand side,
    # 1 item on the right hand side, sorted by lift
    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
    rule_1 = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules)
    rule_2 = filter(lambda rule: len(rule.lhs) == 3 and len(rule.rhs) == 1, rules)


    rule_left = []
    rule_right = []
    # value = []
    rule_body = []
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
      # print(rule)
      rule_left.append(list(rule.lhs))
      rule_right.append(list(rule.rhs))
      rule_body.append([list(rule.lhs), list(rule.rhs), [round(rule.lift,2), round(rule.confidence,2)]])

    for rule in sorted(rule_1, key=lambda rule: rule.lift):
      rule_left.append(list(rule.lhs))
      rule_right.append(list(rule.rhs))
      rule_body.append([list(rule.lhs), list(rule.rhs), [round(rule.lift,2), round(rule.confidence,2)]])
      
      # print(rule) 
    for rule in sorted(rule_2, key=lambda rule: rule.lift):
      rule_left.append(list(rule.lhs))
      rule_right.append(list(rule.rhs))
      rule_body.append([list(rule.lhs), list(rule.rhs), [round(rule.lift,2), round(rule.confidence,2)]])

    
    ac = ['acceptable', 'unacceptable','completely unacceptable', 'completely acceptable', 'neutral' ]
    rule_base = []
    for i in rule_body:
      if len(i[1]) == 1 and i[1][0] in ac:
        if i[2][0] >= 1:
            rule_base.append(i)
            

    return rule_base
    #rule_base[i] = [[ 'voice recording', 'partner', 'with purpose&condition2'], ['Acceptable'], [1.99, 0.99]]





def rules_to_norms(datalist):
    #datalist is a list/dictionary
    
    rows = {}
    for i, item in enumerate(datalist):
        rows[key] = i
        rows['i'] = item
    return rows
    #think about how to select items from rule base
        




def norm_format(conditions, acceptbility, formatted_action, values):
    #norm: modality, precondition, action, effect, confidence
    formatted_norm = ['_', '_', '_', '_', '_']
    if acceptbility == 'acceptable':
        modality = 'P'
    else:
        modality = 'F'

    formatted_norm[0] = modality
    for item in conditions:
        if conditions == 'notified':
            formatted_norm[1] = 'notified'
        elif conditions == 'anonimised':
            formatted_norm[1] = 'anonimised'
        elif conditions == 'confidential':
            formatted_norm[3] = 'confidential'
        elif conditions == 'store':
            formatted_norm[3] ='store'
        elif conditions == 'review':
            formatted_norm[3] = 'review'

    formatted_norm[2] = formatted_action
    formatted_norm[4] = values
    return formatted_norm



def similar_type_1(action1, action_list):
    '''
    This is the first type of method to calculate the similarity between two actions.
    *** How many parameters they have are same ***
    '''

    count_final = []
    similar_summary = []
    similar_for_each_action = []

    for act in action_list:
        count = 0
        for index, act_item in enumerate(act):
            if act_item == action1[index]:
                count += 1
                # import pdb;pdb.set_trace()
        count_final.insert(0,count)
        similar_for_each_action = act + count_final[:1]
        similar_summary.append(similar_for_each_action)
        # import pdb;pdb.set_trace()
  

    similar_summary.sort(key=takeSecond, reverse=True)
    the_most_similar_action = similar_summary[0][0:-1]
    

    return the_most_similar_action
    #return 最similar的action(只有一个)






def euclidean_distance(u, v):
    distance_o = 0
    sum = 0
    for i in range(len(u)):
        a = u[i] - v[i]
        pow_a = math.pow(a,2)
        sum += pow_a
    distance_o = math.sqrt(sum)
    return distance_o


def manhattan_distance(u, v):
    u = np.mat(u)
    v = np.mat(v)
    dist1=float(np.sum(np.abs(u - v)))
    import pdb;pdb.set_trace()
    return dist1





def cosine_similarity(u, v):    
    """
    Cosine similarity reflects the degree of similariy between u and v 
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)
    """
    distance = 0.0

    # Compute the dot product between u and v 
    dot = np.dot(u.T, v)    
    # Compute the L2 norm of u
    norm_u = np.sqrt(np.sum(u**2))    
    # Compute the L2 norm of v 
    norm_v = np.sqrt(np.sum(v**2))    
    # Compute the cosine similarity defined by formula (1)
    cosine_similarity = dot/(norm_u * norm_v)    
    return cosine_similarity


# father = embeddings_index['father']
# mother = embeddings_index['mother']
# print(cosine_similarity(father, mother))
# print(cosine_similarity(mother,father))
# import pdb;pdb.set_trace()


def takeSecond(elem):
    return elem[7]


def element_similarity(elem1, elem2):
    # elem1 = 'healthcare_data'
    # elem2 = '_'
    count_list = []
    if elem1.strip().lstrip() == '_' or elem2.strip().lstrip() == '_':
        count = 1
    else:
        elem1 = '_'.join(elem1.split())
        elem2 = '_'.join(elem2.split())
        elem1 = elem1.split('_')
        elem2 = elem2.split('_')
        elem1_emb = np.array([0] * len(embeddings_index['the']), dtype='float32')
        elem2_emb = np.array([0] * len(embeddings_index['the']), dtype='float32')
        # Approach1: Addition
        for order, x in enumerate(elem1):
            elem1_emb += embeddings_index[x]
        for order, x in enumerate(elem2):
            elem2_emb += embeddings_index[x]
        count = cosine_similarity(elem1_emb, elem2_emb)

        # Approach2: Elementwise max
        # elem1_emb_array = []
        # elem2_emb_array = []
        # for order, x in enumerate(elem1):
        #     elem1_emb_array.append(embeddings_index[x])
        # for order, x in enumerate(elem2):
        #     elem2_emb_array.append(embeddings_index[x])
        # elem1_emb = np.max(np.stack(elem1_emb_array), axis=0)
        # elem2_emb = np.max(np.stack(elem2_emb_array), axis=0)
        # count = cosine_similarity(elem1_emb, elem2_emb)

        # Approach3: Max similarity of word pairs
        # for order, x in enumerate(elem1):
        #     x_count = []
        #     for ordery, y in enumerate(elem2):
        #         x_count.append(cosine_similarity(embeddings_index[x],embeddings_index[y]))
        #     count_list.append(max(x_count))
        # count = sum(count_list) / len(count_list)

        # Approach4: Average [this one is not correct]
        # for order, x in enumerate(elem1):
            # for ordery, y in enumerate(elem2):
                # count_list.append(cosine_similarity(embeddings_index[x],embeddings_index[y]))
        # count = sum(count_list) / len(count_list)
    return count


embeddings_index = {}
f = open('glove.6B.100d.txt')   
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()



def find_most_similar_action(action_list, action1):
    count_final = []
    # action1 = ['send', 'spa', '_', 'weather_forecast', 'weather', 'primary_user', '_']


    # embeddings_index = {}
    # f = open('glove.6B.100d.txt')   
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()

    for m, act in enumerate(action_list):
        similar = []
        for index, act_item in enumerate(act): #两个action之间所有的的element pair都要计算  act_item = 'healthcare_data'
            similar.append(element_similarity(act_item, action1[index]))
        act.append(sum(similar) / len(similar))
        count_final.append(act)



    count_final.sort(key=takeSecond, reverse=True)
    most_similar = count_final[0][0:-1]
    return most_similar






def two_actions_are_same(action1, action2):
    # list1 = ["one","two","three"]
    # list2 = ["one","three","two"]
    # list1 = sorted(list1)
    # list2 = sorted(list2) # don't change list itself
    # print(list1 == list2)
    if action1 == action2:
        return 'True'
    else:
        return 'False'



def action_determine(action):
    #e.g.: action = ['send', 'spa', 'other skills', 'to_do', 'to do list', 'primary_user', '_']
    key_word_data = action[4]

    anonimised_datas = []
    notified_datas = []

    # import pdb;pdb.set_trace()
    if key_word_data != '_':
        for item in knowledge_base[key_word_data]['anonimised']:
            if item == action[5]:
                anonimised_datas.append(item)
        for item in knowledge_base[key_word_data]['notified']:
            if item == action[5]:
                notified_datas.append(item)
    
    '''
    First: find the relevant preconditions in the knowledge base;
            If exist: find active norms in the norm base
                Active norms base:
                    If a same action exists, return the modality as result
                    If no same action exists, find the most similar one
            If not exist: find all the norms in the norm base
                Norm base: 
                    If a same action exists, return the modality as result
                    if no same action exists, find the most similar one
            When check all the norms in the norm base, remember to check the most general one
    '''
    active_norm_base = {}  #actions are keys, modality and confidence are values

    #如果precondition在knowlege base中已经存在， 那么将active norm放入active norm base中
    #anonimised_datas= ['primary_user','bob']
    empty = []
    if anonimised_datas != empty:
        for item in anonimised_datas:
            if norm_base[key_word_data]['anonimised'][item]:
                for keys, values in norm_base[key_word_data]['anonimised'][item].items():
                    active_norm_base[keys] = values
    if notified_datas != empty:
        for item in notified_datas:
            if norm_base[key_word_data]['notified'][item]:
                for keys, values in norm_base[key_word_data]['notified'][item].items():
                    active_norm_base[keys] = values
    if norm_base[key_word_data]['_']['_'] != {}:
        for keys, values in norm_base[key_word_data]['_']['_'].items():
            active_norm_base[keys] = values

    
    #判断active_norm_base是否为空
    possible_actions = []
    result = []
    if active_norm_base != {}:
        for item in active_norm_base.keys():
            l = list(item)
            if l == action:
                result.append(list(active_norm_base[item].keys())[0])
                #active norm base中有相同的action 直接返回result
            else:
                possible_actions.append(l)
        if result != empty:
            return result[0]
        else:
            most_similar_action = find_most_similar_action(possible_actions,action)
            x = tuple(most_similar_action)
            result.append(list(active_norm_base[x].keys())[0])
            return result[0]
            #找similar的norm
    else:
        #active norm base为空
        #先找general的norm
        #再找剩下norm中最相似的
        matching_norm_base = {}
        possible_actions_1 = []
        for key, value in norm_base['_'].items():
            for key1, value1 in value.items():
                for key2, value2 in value1.items():
                        matching_norm_base[key2] = value2
                        possible_actions_1.append(list(key2))
                        # import pdb;pdb.set_trace()
        most_similar_action = find_most_similar_action(possible_actions_1,action)
        x = tuple(most_similar_action)
        result.append(list(matching_norm_base[x].keys())[0])
        return result[0]





def takeSecond(elem):
    '''
    Take second element for sort
    '''
    return elem[1]

def takeSeventh(elem):
    '''
    Take sixth element for sort
    '''
    return elem[7]


def str_to_list(item):
    #"['thermostat', 'law enforcement agencies', 'with purpose&condition5', '_', 'somewhat acceptable']"
    x = len(item)
    y = item[1:x-1].replace("'","")
    new_list = y.split(', ')
    return new_list

def load_test(datatype):
    data_file = data_file = './data/sub_data/' + datatype + '_test.csv'
    with open(data_file) as f:
        reader = csv.reader(f)
        all_data = []
        for rows in reader:
            for index , item in enumerate(rows):
                rows[index] = str_to_list(item)
            all_data.append(rows)
        return all_data


if __name__ == "__main__":

    # all_data = data_input('./data/new_file.csv')
    for index, item in enumerate(data_use_to_create_base['data'][:-1]):
        if index == 4:

            # (training_set, test_set) = sample_split(all_data, item)
            # # all_data = data_trans_to_plaintext('./data/new_file.csv')   #all the data in plain text format
            # x = to_plaintext(training_set)
            test_set = load_test(item)
            initial_rules_1 = rule_mining(0.015, 0.56, item)
            initial_rules_2 = []
            exception = ['with purpose&no condition', 'no purpose&no condition', 'with purpose&condition1','with purpose&condition2','with purpose&condition3','with purpose&condition5','with purpose&condition4']
            for k, v in enumerate(initial_rules_1):
                if len(v[0]) == 1 and v[0][0] not in exception:
                    initial_rules_2.append(v)
                elif len(v[0]) > 1:
                    initial_rules_2.append(v)

           
            '''
                Build Norm Base:
                Norm Base is a dictionary and can be traced by 'data' -> 'precondition' -> 'effect' -> 'action' -> 'modality + value'    
            '''

            norm_build_dict = {
                    '_': {
                        '_': {},
                        'confidential': {},
                        'review': {},
                        'store': {}
                        },
                    'anonimised': {},
                    'notified': {}
                    }
            norm_base = {data:copy.deepcopy(norm_build_dict) for data in data_use_to_create_base['data']}

            #create same data structure for matching norms and active norms
            # matching_norms = copy.deepcopy(norm_base)
            # active_norms = copy.deepcopy(norm_base)

            
            norm_collection = []
            action_collection = []
            #测试用 initial_rules
            for index, item in enumerate(initial_rules_2):
            #for item in initial_rules:
                #[['skills', 'with purpose&no condition'], ['completely unacceptable'], [1.59, 0.65]]
                # e.g.: item[0] = ['call assistant', 'prime user', 'with the purpose of knowing the data']
                # e.g.: item[1] = ['Acceptable']
                # e.g.: item[2] = [1.2, 1.5]  lift & confidence
                (action, condition_type) = action_format(item[0])
                #action = ['send', 'spa', 'neighbours', 'ride_service', 'location', 'primary_user', 'with the purpose of knowing the data']
                norm = norm_format(condition_type, item[1][0], action, item[2])
                # import pdb;pdb.set_trace()
                # norm = [modality, precondition, action, effect, confidence]
                norm_collection.append(norm)
                action_collection.append(action)
                import pdb;pdb.set_trace()





                if norm[1] == '_': #precondition
                    # norm_base[action[4]][norm[1]][norm[3]] = {tuple(action):{norm[0]:norm[4]}}
                    if tuple(action) not in norm_base[action[4]][norm[1]][norm[3]]:
                        norm_base[action[4]][norm[1]][norm[3]][tuple(action)] = {}
                    if norm[0] not in norm_base[action[4]][norm[1]][norm[3]][tuple(action)]:
                        norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]] = [norm[4]]
                        if len(norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]]) > 1:
                            norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]].sort(key=takeSecond, reverse=True)
                            norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]] = norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]][0]
                    else:
                        if norm[4] not in norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]]:
                            norm_base[action[4]][norm[1]][norm[3]][tuple(action)][norm[0]].append(norm[4])
                else:   #precondition = anonimised or notified
                    if action[5] not in norm_base[action[4]][norm[1]]:
                        norm_base[action[4]][norm[1]][action[5]] = {}
                    if tuple(action) not in norm_base[action[4]][norm[1]][action[5]]:
                        norm_base[action[4]][norm[1]][action[5]][tuple(action)] = {}
                    if norm[0] not in norm_base[action[4]][norm[1]][action[5]][tuple(action)]:
                        norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]] = [norm[4]]
                    else:
                        if norm[4] not in norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]]:
                            norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]].append(norm[4])
                        if len(norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]]) > 1:
                            norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]].sort(key=takeSecond, reverse=True)
                            norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]] = norm_base[action[4]][norm[1]][action[5]][tuple(action)][norm[0]][0]
            
            
            '''
                Defaults is the rules mind using the training set
                and will be put into the norm base
            '''
            defaults = copy.deepcopy(norm_base)
            import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace() 

           

            '''
            ***************Feature1: Manage knowledge base**********************
            knowledge base only keeps the preconditions hold by the SPA
            '''


            # import pdb;pdb.set_trace()
            knowledge_base = {}
            for datatype, subdict in defaults.items():
                if datatype != '_':
                    knowledge_base[datatype] = {'anonimised':[], 'notified': []}
                    for key1, value1 in subdict.items():
                        if key1 in ['anonimised', 'notified'] and value1 != {}:
                            for key2, value2 in value1.items():
                                knowledge_base[datatype][key1].append(key2)
                     
            #检查knowledge base中是否有相应的precondition 如果有返回True 如果没有返回False
            final_result = []

            for item in test_set:
                correct_count = 0
                if len(item) == 0:
                    import pdb;pdb.set_trace()
                accu_result = []
                for index, value in enumerate(item):
                    act = value[:-2]
                    (new_act, this_condition) = action_format(act)
                    result = action_determine(new_act)
                    accuracy = accuracy_verify(result, value[4])
                    accu_result.append(accuracy)
                correct_count += accu_result.count('1') 
                # final = accu_result.count('1') / len(accu_result)
                # final_result.append(final)
                # print(final)
                final_result.append(correct_count/len(item))
                # import pdb;pdb.set_trace() 
                # print('accuracy:', correct_count/len(item))
            print('total:', np.mean(final_result))
            import pdb;pdb.set_trace() 
        






    #Knowledge base : add and remove

    #Knowledge base: When new actions are monitored
    #new actions look like: (email, parents, no purpose&no condition, prime user,Acceptable)

    # new_action_list = []

    # for i in test_set:
    #     #item = ['healthcare', 'advertising agencies', 'with purpose&no condition', 'prime user', 'Acceptable']
    #     #i[0] = data
    #     #i[3] = subject
    #     n = [i[0], i[1], i[2], i[3]]
    #     #i[4] = acceptable
    #     (new_act_monitored, new_act_condition) = action_format(n)
    #     new_norm_from_new_act = norm_format(new_act_condition, i[4], new_act_monitored, [])
    #     #new_norm_from_new_act is used to compare user perceptions(new_norm_from_new_act[0]) with NSA results
    #     new_action_list.append(new_act_monitored)
    #     import pdb;pdb.set_trace()


    # for i in new_action_list:
    #     # import pdb;pdb.set_trace()
    #     result = action_determine(norm_base, knowledge_base, i)

        




    
    


    


































