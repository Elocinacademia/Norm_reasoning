import csv 
from collections import defaultdict
from pprint import pprint
import pandas as pd
import re
import copy
import random

rangking_re = {'your parents':1, 
'your partner':2, 
'your siblings':3, 
'your housemates':4, 
'your children':5, 
'neighbours':6, 
'close friends':7,
'your friends':7, 
'close family':8, 
'house helper/keeper':9,
'house hepler/keeper':9, 
'house keeper/helper':9, 
'visitors in general':10, 
'assistant provider':11,
'skills':12,
'other skills':13,
'advertising agencies':14,
'law enforcement agencies':15}

acceptable_scale = {'completely acceptable': 'acceptable',
'completely acceptable ': 'acceptable',
'somewhat acceptable': 'acceptable',
'neutral' : 'acceptable',
'somewhat unacceptable': 'unacceptable',
'completely unacceptable': 'unacceptable',
'click to write scale point 6': 'acceptable'
}

random.seed(1)

ecoquestion = [14,23,32,41,50,60,69,78,87,96,106,115,124,133,142]

datatype = {'email':[12,13,14,15,16,17,18,19],
'banking': [21,22,23,24,25,26,27,28],
'healthcare': [30, 31,32,33,34,35,36,37],
'door locker':[39,40,41,42,43,44,45,46],
'camera': [48,49,50,51,52,53,54,55],
'call assistant': [58,59,60,61,62,63,64,65],
'video call': [67,68,69,70,71,72,73,74],
'location': [76,77,78,79,80,81,82,83],
'voice recording': [85,86,87,88,89,90,91,92],
'todo': [94,95,96,97,98,99,289,290],
'sleep hours': [104,105,106,107,108,109,110,111],
'playlists': [113,114,115,116,117,118,119,120],
'thermostat': [122,123,124,125,126,127,128,129],
'shopping': [131,132,133,134,135,136,137,138],
'weather': [140,141,142,143,144,145,146,147]}

skillstype = {'skills': [16,25,34,43,52,62,71,80,89,98,108,117,126,135,144],
'other skills': [17,26,35,44,53,63,72,81,90,99,109,118,127,136,145]}


data_use_to_create_base = {'data': ['email', 'banking', 'healthcare', 'door locker', 'call assistant', 'video call', 'location', 'voice recording', 'playlists', 'thermostat', 'shopping', 'weather', '_']}
'''
no 'smart camera', 'to do list' and 'sleeping hours',
because in our dataset, all users answered questions in cluded these three datatypes.
Thus we are not able to test whethet similarity fucntion works on these new added datatypes
'''



reverse_datatype = {}
for key, value in datatype.items():
    for question in value:
        reverse_datatype[question] = key

reverse_skillstype = {}
for key, value in skillstype.items():
    for question in value:
        reverse_skillstype[question] = key

reverse_recipientstype = {}
for key, value in rangking_re.items():
        reverse_recipientstype[value] = key




        


'''
1. ecoquestion example: Assume that you manage your bank accounts through a voice assistant e.g. Amazon Echo/ Alexa, Google Home/ Assistant. 
                        How acceptable is it for your bank account details to be shared with the following recipients: 


2. Skills and other skills type example: Assume that you manage your bank accounts through a voice assistant, e.g. Amazon Echo/Alexa, Google Home/ Assistant. 
How acceptable is it for your bank account details to be shared with providers of Skills or Actions in the Business & Finance category (e.g. PayPal) for making transactions, 
under the following conditions: 


3. Other normal types: Assume that you manage your bank accounts through a voice assistant e.g. Amazon Echo/Alexa, Google Home/ Assistant. 
                        How acceptable is it for your bank account details to be shared with law enforcement agencies for the purpose of 
                        investigating a crime, under the following conditions: 

'''


f = open('./data/data_ques_pure.csv')
reader = csv.reader(f)
header_row = next(reader)


file = pd.read_csv('./data/data_ques_1.csv')
ques_number = []
ques_number = file.iloc[0].values.tolist()
question_collection = []
question_collection = [i for i in file.columns]




count = 0

info_flow = []
question = []
# this_flow = [0,0,0,0,0]
info_flow_dic = {}

#(email, parents, no purpose&no condition, prime user,Acceptable)

for number,row in enumerate(reader):
    count += 1
    info_flow = []
    # import pdb; pdb.set_trace()
    for index, item in enumerate(row):
        if item != 'empty' and '_' not in item:
            this_flow = []
            question.append(header_row[index])
            question.append(item)
            question_text = question[0]
            question_text = question_text = ' '.join(question_text.split())
            m = ques_number[index].split('_')[0]    #question number eg:Q31
            m = m[1:]
            m_sub = ques_number[index].split('_')[1]   #sub question number eg:1
            # import pdb; pdb.set_trace()
            if m in ecoquestion:
                collection = []
                this_datatype = reverse_datatype[m]
                this_purpose = 'no purpose'
                this_condition = 'no condition'
                # this_recipient = question[0].split('-')[1]
                if m_sub == '2':
                    this_recipient = 'third party skill'
                elif m_sub == '3':
                    this_recipient = 'other skills'
                else:
                    this_recipient = question_text[0].split(' - ')[-1].lower()
            elif m in skillstype['skills']:
                this_datatype = reverse_datatype[int(m)]
                this_recipient  = 'third party skill'
                this_purpose = 'with purpose'
                this_condition = question_text[0].split(' - ')[-1].lower()
                # import pdb; pdb.set_trace()
            elif m in skillstype['other skills']:
                this_datatype = reverse_datatype[int(m)]
                this_recipient  = 'other skills'
                this_purpose = 'with purpose'
                this_condition = question_text[0].split(' - ')[-1].lower()
            else:
                this_datatype = reverse_datatype[int(m)]
                recipient_score = None
                if 'following conditions' in question_text.split(' - ')[0].lower():
                    this_condition = question_text.split(' - ')[1].lower()
                    this_purpose = 'with purpose'
                    for key, value in rangking_re.items():
                        if key in question_text.split(' - ')[0].lower() and recipient_score is None:
                            recipient_score = value
                            this_recipient = reverse_recipientstype[value]
                            # this_recipient = reverse_recipientstype[value]
                        elif key in question_text.split(' - ')[0].lower() and recipient_score is not None:
                            raise Exception('Multiple recipient type found')
                    if recipient_score == None:
                        raise Exception('Recipient type not found')
                else:
                    this_condition = 'no condition'
                    if 'for the purpose' in question_text.split(' - ')[0].lower():
                        this_purpose = 'with purpose'
                    else:
                        this_purpose = 'no purpose'
                    for key, value in rangking_re.items():
                        if key in question_text.split(' - ')[-1].lower() and recipient_score is None:
                            recipient_score = value
                            this_recipient = reverse_recipientstype[value]
                        elif key in question_text.split(' - ')[0].lower() and recipient_score is not None:
                            raise Exception('Multiple recipient type found')
                        # import pdb; pdb.set_trace()
                    if recipient_score == None:
                        raise Exception('Recipient type not found')
                # import pdb; pdb.set_trace()
            this_flow.append(this_datatype)
            # this_flow.append(recipient_score)
            if this_recipient == 'house keeper/helper':
                this_recipient = 'house keeper'
            this_flow.append(this_recipient)
            # this_flow.append(this_purpose)
            # this_flow.append(this_condition)
            #combine this condition and this purpose
            # import pdb; pdb.set_trace()
            # print(this_purpose)
            # print(this_condition)

            if this_purpose == 'no purpose' and this_condition in ['no condition', 'no conditions']:
                this_principle = 'no purpose&no condition'
            elif this_purpose == 'with purpose' and this_condition in ['no condition', 'no conditions']:
                this_principle = 'with purpose&no condition'
            elif this_purpose == 'with purpose' and 'notified' in this_condition:
                this_principle = 'with purpose&condition1'
            elif this_purpose == 'with purpose' and 'anonymous' in this_condition:
                this_principle = 'with purpose&condition2'
            elif this_purpose == 'with purpose' and 'confidential' in this_condition:
                this_principle = 'with purpose&condition3'
            elif this_purpose == 'with purpose' and 'store' in this_condition:
                this_principle = 'with purpose&condition4'   
            elif this_purpose == 'with purpose' and 'review' in this_condition:
                this_principle = 'with purpose&condition5'  
            else:
                print('we have something wrong:',this_condition,this_purpose)

            this_flow.append(this_principle)
            this_flow.append('_')
            if item.lower() in acceptable_scale.keys():
                this_item = acceptable_scale[item.lower()]

            # this_flow.append(acceptable_scale[this_item])
            this_flow.append(this_item)
            
            info_flow.append(this_flow)
            # import pdb; pdb.set_trace()
                # this_flow[0] = this_datatype
                # this_flow[1] = recipient_score
                # this_flow[2] = this_purpose
                # this_flow[3] = this_condition
                # this_flow[4] = item    
        question.clear()
    info_flow_dic[number+1] = info_flow
    # if number == 3:
    #     import pdb; pdb.set_trace()
        



new_file = './data/new_file.csv'
with open(new_file,"w") as csv_file:
    writer=csv.writer(csv_file)
    for key,value in info_flow_dic.items():
        # import pdb; pdb.set_trace() 
        writer.writerow(value)


# def to_plaintext(training_set):
#     with open('./data/plain_data.csv','w') as fout:
#         writer = csv.writer(fout)
#         for rows in training_set:
#             for item in rows:
#                 if '_' not in item:
#                     import pdb; pdb.set_trace()
#                     print(item)
#                 item.remove('_')
#                 writer.writerow(item)
                 


def data_input(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        all_data = []
        for rows in reader:
            for index , item in enumerate(rows):
                rows[index] = str_to_list(item)
            all_data.append(rows)
        return all_data

def str_to_list(item):
    #"['thermostat', 'law enforcement agencies', 'with purpose&condition5', '_', 'somewhat acceptable']"
    x = len(item)
    y = item[1:x-1].replace("'","")
    new_list = y.split(', ')
    return new_list


def sample_split(data, target_datatype):
    ''''email', 'banking', 'healthcare', 'door locker', 'smart camera', 'camera', 'todo', 'call assistant', 'video call', 'location', 'voice recording', 'to do list', 'sleeping hours', 'sleep hours', 'playlists', 'thermostat', 'shopping', 'weather
    This is used to split the sample
    Return training set for rule mining and base development
    '''

    #seed = 'email'
    # if target_datatype == 'smart camera':
    #     import pdb; pdb.set_trace()
    collections = []
    others = []
    for row in data:
        datatype = []
        for item in row:
            if item[0] not in collections:
                datatype.append(item[0])
        # import pdb; pdb.set_trace()
        if target_datatype in datatype:
            collections.append(copy.deepcopy(row))
        else:
            others.append(copy.deepcopy(row))
    training_set = others
    random.shuffle(collections)
    cut = round(len(others)/4)
    test_set = collections[:cut]
    print(len(training_set), target_datatype)
    #'./data/plain_data.csv'

    file1 = './data/sub_data/' + target_datatype + '_train.csv'
    file2 = './data/sub_data/' + target_datatype + '_test.csv'
    
    with open(file1,'w') as fout:
        writer = csv.writer(fout)
        for rows in training_set:
            for item in rows:
                if '_' not in item:
                    print(item)
                item.remove('_')
                writer.writerow(item)

    with open(file2,'w') as fout:
        writer = csv.writer(fout)
        for rows in test_set:
            writer.writerow(rows)

    return training_set, test_set



all_data = data_input('./data/new_file.csv')
for item in data_use_to_create_base['data'][:-1]:
    (training_set, test_set) = sample_split(all_data, item)
    # all_data = data_trans_to_plaintext('./data/new_file.csv')   #all the data in plain text format
    # x = to_plaintext(training_set)   
      







                




            





















