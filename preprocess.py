import json
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from sklearn import svm

from sklearn.feature_extraction.text import CountVectorizer

# tokenizer = TreebankWordTokenizer()
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer() 

# with open('train.json','r') as f:
#     text = f.read()
#     text = json.loads(text)

# f.close()
# # print(text['train-0'])

# train_set = []
# for i,t in text.items():
#     train_set.append(t['text'])
#     # print(i,t['text'])

# with open('dev.json','r') as f:
#     dev = f.read()
#     dev = json.loads(dev)
# f.close()

# dev_set = []
# dev_label = []
# for i,t in dev.items():
#     dev_set.append(t['text'])
#     dev_lable.append(t['label'])
    
# print(type(train_set[0]))

def preprocessing(text):
    text = text.replace('{html}',"")
    rem_url = re.sub(r'http\S+', '', text)
    tokens = tokenizer.tokenize(rem_url)
    rem_stop_words  = [w for w in tokens if len(w)>2 
        if not w in stopwords.words('english')]
    # stem_words = [stemmer.stem(w) for w in rem_stop_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in rem_stop_words]
    # print(text)

    # print('-----------------')
    # print(lemma_words)
    return lemma_words



print(99/2)
 
