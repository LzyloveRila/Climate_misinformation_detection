import json
import nltk
# from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 


with open('train.json','r') as f:
    text = f.read()
    text = json.loads(text)
f.close()

data = []
for i,t in text.items():
    data.append(t['text'])

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
    tokens = nltk.word_tokenize(rem_url)
    tokens = [w.lower() for w in tokens]
    rem_stop_words  = [w for w in tokens if len(w)>2 
        if not w in stopwords.words('english')]
    # stem_words = [stemmer.stem(w) for w in rem_stop_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in rem_stop_words]
    # print(text)

    # print('-----------------')
    # print(lemma_words)
    return " ".join(lemma_words)#lemma_words

def print_top_words(model, feature_names, n_top_words):
    # print the term with higher weight in the topic
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    #print topic-word matrix
    print(model.components_)

def lda_sklearn():
    train_set = [preprocessing(text) for text in data[:5]]
    test_set = [preprocessing(text) for text in data[6:10]]
    # print(test_set)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,stop_words='english')
    # max_features=n_features,
    tf = tf_vectorizer.fit_transform(train_set)
    vectorizer_test = CountVectorizer(max_df=0.95, min_df=2,stop_words='english',vocabulary=tf_vectorizer.vocabulary_)
    tf_test = vectorizer_test.fit_transform(test_set) 

    n_topic = 2
    lda = LatentDirichletAllocation(n_components=n_topic, max_iter=50,learning_method='batch')
    lda.fit(tf)        

    for doc in tf_test:
        doc_scores = lda.transform(doc)
        print(doc_scores)

    n_top_words=20
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

lda_sklearn()


# refiltered =nltk.pos_tag(train_set[0])
# print(refiltered)

def word2vec():
    from gensim.models import Word2Vec
    documents = ["The cat sat on the mat.", "I love green eggs and ham."]
    sentences = []
    # remove punkt
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for doc in documents:
        doc = re.sub(stop, '', doc)
        sentences.append(doc.split())
    #sentences = [["The", "cat", "sat", "on", "the", "mat"], 
    #            ["I", "love", "green", "eggs", "and", "ham"]]


    # size-embedding size，window-ngram length，workers-num of process
    # ignore the word with frequency less than min_count
    # sg=1-Skip-Gram，otherwise CBOW
    model = Word2Vec(sentences, size=5, window=1, min_count=1, workers=4, sg=1)
    print(model.wv['cat'])