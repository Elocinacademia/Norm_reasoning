

'''
import the libraries and download ‘punkt‘
Natural Language Toolkit，自然语言处理工具包，是NLP研究领域常用的一个Python库
'''

import nltk
# nltk.download('punkt')
# nltk.download()
nltk.word_tokenize('a pivot is a pin fsdfsdfs')
import pdb; pdb.set_trace()
from nltk.tokenize import word_tokenize
import numpy as np


'''
Then, we define our list of sentences. 
You can use a larger list 
(it is best to use a list of sentences for easier processing of each sentence)
'''
sentences = ["I ate dinner.", 
       "We had a three-course meal.", 
       "Brad came to dinner with us.",
       "He loves fish tacos.",
       "In the end, we all felt like we ate too much.",
       "We all agreed; it was a magnificent evening."]


'''
keep  a tokenized version of these sentences
'''
#Tokenization of each document
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))

print(tokenized_sent)


