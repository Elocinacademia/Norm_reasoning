action_list = [['send', 'spa', 'skills', 'call_assistant', 'call assistant', 'primary_user', '_'], ['send', 'spa', '_', 'call_assistant', 'call assistant', 'primary_user', '_'], ['send', 'spa', '_', 'call_assistant', 'call assistant', 'primary_user', 'with the purpose of knowing the data']]


import copy

def form_sentence(ori_action):
    # sentence = ''
    action = copy.deepcopy(ori_action)
    if '_' in action[2]:
        action[2] = ' '.join(action[2].split('_'))
    if '_' in action[4]:
        action[4] = ' '.join(action[4].split('_'))
    if '_' in action[5]:
        action[5] = ' '.join(action[5].split('_'))
    if '_' in action[6]:
        action[6] = ' '.join(action[6].split('_'))
     
    
    if action[2] == ' ' and action[6] == ' ':
        #no recipient no purpose: ('send', 'spa', '_', 'voice_recordings', 'voice recording', 'primary_user', '_')
        sentence = 'The assistant is sending ' + action[4] + ' of ' + action[5] + ' without stating purpose'
    elif action[2] == ' ' and action[6]!= ' ':
        #no recipient: ('send', 'spa', '_', 'voice_recordings', 'voice recording', 'primary_user', '_')
        sentence = 'The assistant is sending ' + action[4] + ' of ' + action[5] + ' ' + action[6]
    elif action[2] != ' ' and action[4] == ' ':
        if action[6] == ' ':
            sentence = 'The assistant is sending the information of ' + action[5] + ' to ' + action[2] + ' without stating purpose'
        else:
            sentence = 'The assistant is sending the information of ' + action[5] + ' to ' + action[2] + ' ' + action[6]
    else:
        if action[6] == ' ':
            sentence = 'The assistant is sending ' + action[4] + ' of ' + action[5] + ' to ' + action[2]+ ' without stating purpose'
        else:
            sentence = 'The assistant is sending ' + action[4] + ' of ' + action[5] + ' to ' + action[2]+  ' ' + action[6]
    
    

    return sentence





sentence_collection = []
for items in action_list:
    result = form_sentence(items)
    sentence_collection.append(result)
import pdb; pdb.set_trace()  




