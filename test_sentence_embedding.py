

'''
import the libraries and download ‘punkt‘
Natural Language Toolkit，自然语言处理工具包，是NLP研究领域常用的一个Python库
'''


import nltk

'''
As we download the nltk packages ourselve,
we need to bypass the ssl check
'''
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()



nltk.download('punkt')
# nltk.download()

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
# import pdb; pdb.set_trace()
print(tokenized_sent)



'''
Define a function which returns the cosine similarity between 2 vectors
'''
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# '''
# ---------------------------------------------------------------------------------------------------------------------------------
# Doc2Vec
# Step 1:
# We will use Gensim to show an example of how to use Doc2Vec. 
# Further, we have already had a list of sentences. 
# We will first import the model and other libraries and then we will build a tagged sentence corpus. 
# Each sentence is now represented as a TaggedDocument containing a list of the words in it and a tag associated with it.
# '''

# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]

# print(tagged_data)

# '''
# We then train the model with the parameters:
# '''
# model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)

# '''
# vector_size = Dimensionality of the feature vectors.
# window = The maximum distance between the current and predicted word within a sentence.
# min_count = Ignores all words with total frequency lower than this.
# alpha = The initial learning rate.
# '''

# ## Print model vocabulary
# # model.wv.vocab
# # model.wv.key_to_index



# '''
# Step 3:
# We now take up a new test sentence and find the top 5 most similar sentences from our data. 
# We will also display them in order of decreasing similarity. 
# The infer_vector method returns the vectorized form of the test sentence(including the paragraph vector). 
# The most_similar method returns similar sentences
# '''

# test_doc = word_tokenize("sugar reading".lower())
# test_doc_vector = model.infer_vector(test_doc)
# # model.docvecs.most_similar(positive = [test_doc_vector])
# kk = model.dv.most_similar(positive = [test_doc_vector])


# ###positive = List of sentences that contribute positively.

# ---------------------------------------------------------------------------------------------------------------------------------

'''
BERT
We will then load the pre-trained BERT model. 
However, there are many other pre-trained models available. 
'''



from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

'''
Step 2:
We will then encode the provided sentences. 
We can also display the sentence vectors(just uncomment the code below)
'''
import pdb; pdb.set_trace()

sentence_embeddings = model.encode(sentences)

#print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
#print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])


'''
Step 3:
Then we will define a test query and encode it as well:

'''

query = "I had pizza and pasta"
query_vec = model.encode([query])[0]

'''
Step 4:
We will then compute the cosine similarity using scipy. 
We will retrieve the similarity values between the sentences and our test query:
'''

for sent in sentences:
  sim = cosine(query_vec, model.encode([sent])[0])
  print("Sentence = ", sent, "; similarity = ", sim)



'''
Additional comments:
There you go, we have obtained the similarity between the sentences in our text and our test sentence. 
A crucial point to note is that SentenceBERT is pretty slow if you want to train it from scratch.
'''


import pdb; pdb.set_trace()




