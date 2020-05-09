import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer
# from textblob import TextBlob
import nltk
from tqdm import tqdm
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def real_data_to_trainjson():
    df = pd.read_csv('./new data/all_data.csv',index_col=0)
    # print(df)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    truth_set = {}
    count = 2762
    for row in df.itertuples():
        # print(row[21])  #18 text  21 type
        if row[21] == 'real':
            key_name = 'train-' + str(count)
            # print("text:",row[18])
            tokened_text = tokenizer.tokenize(row[18])
            if len(tokened_text)<1100 and len(tokened_text)> 25:
                truth_set[key_name] = {"text":row[18],'label':0}
                count += 1
            if count == 4000:
                break

    with open('train_fact.json','w') as f:
        json.dump(truth_set,f)

real_data_to_trainjson()

def chechk_len():
    with open('tiny_set.json','r') as f:
        text = json.load(f)
    train_set = []
    for i,t in text.items():
        train_set.append(t['text'])
    print(len(train_set)) #1168 pos      8074 neg


def concatenate_pos_neg_set():
    with open('train_climate.json','r') as f1:
        text1 = json.load(f1)
    f1.close()
    with open('train_climate_all.json','r') as f2:
        text2 = json.load(f2)
    f2.close()

    dataset = {**text1,**text2}
    with open('train_climate_news.json','w') as f:
        json.dump(dataset,f)

# concatenate_pos_neg_set()

# train_data = []
# # train_labels = []
# with open('train_1100.json','r') as f:
#   data = json.load(f)
#   for v in data.values():
#     s = '[CLS] ' + v['text'] + ' [SEP]'
#     train_data.append(s)
#     # train_labels.append(v['label'])
# f.close()


def length_count(train_data):
    print("length_dataset:",len(train_data))
    # print(train_data[0])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    tokenized_train=[tokenizer.tokenize(sent) for sent in train_data] 

    length = [len(i) for i in tokenized_train]
    # print(length)

    print('-------------------------')
    print("average:",np.mean(length))
    print('min:',np.min(length))
    print("max:",np.max(length))
    x_128 = 0
    x_256 = 0
    x_512 = 0
    total = len(train_data)
    x_1024 = 0
    for i in length:
        if i<=128:
            x_128+=1
        if i<= 256:
            x_256+=1
        if i<=512:
            x_512+=1
        if i<=1024:
            x_1024+=1
    print("xxx128:",x_128,"percent:{:.2%}".format(x_128/total))
    print("xxx256:",x_256,"percent:{:.2%}".format(x_256/total))
    print("xxx512:",x_512,"percent:{:.2%}".format(x_512/total))
    print("x_1024:",x_1024,"percent:{:.2%}".format(x_1024/total))

def truncating_from_middle(input_lists,maxlen,value=0):
    half = int(maxlen/2)
    new_lists = []
    for l in input_lists:
        if len(l) > maxlen:
            print(len(l),half)
            post = (len(l)-half)
            new_l = l[:half] + l[post:]
            new_lists.append(new_l)
        else:
            pad_need = maxlen-len(l)
            l = l + [0] * pad_need
            new_lists.append(l)
    return new_lists

def sentiment():
    # find top sent in train_1100
    with open('train_climate.json','r') as f:
        text = f.read()
        text = json.loads(text)

    f.close()
    # print(text['train-0'])

    train_total_set = []
    train_label_1100 = []
    for i,t in text.items():
        train_total_set.append(t['text'])
        train_label_1100.append(t['label'])

    # analyser = SentimentIntensityAnalyzer()
    # pos : compound > 0.05
    # neg : compound < -0.05
    # neu : [-0.05,0.05]

    # subjectivity 0-1
    neu,pos,neg = 0,0,0

    # for sent in sent1:
    subj_lists = []
    objective = 0
    subjevt = 0
    for i in tqdm(train_total_set):
        sentences = nltk.sent_tokenize(i)
        avg_subjectivity = []
        obj,subj = 0,0
        for sent in sentences:
            doc2 = TextBlob(sent)
        # score = analyser.polarity_scores(i)
        # score['compound]
        # polarity = doc2.polarity
        # if polarity > 0.0:
        #     pos+=1
        # elif polarity < 0.0:
        #     neg+=1
        # else:
        #     neu+=1
            subjectivity = doc2.subjectivity
            avg_subjectivity.append(subjectivity)
            if subjectivity < 0.3:
                obj += 1
            else:
                subj +=1
        # print(np.mean(avg_subjectivity))
        # print("obj:",obj," subj:",subj)
        subj_lists.append(np.mean(avg_subjectivity))
        if np.mean(avg_subjectivity) > 0.3:
            subjevt += 1
        else:
            objective +=1 
    print(subjevt,objective)


    
    
    # print("pos:",pos," neg:",neg," neu:",neu)
    # print(sent1)
    # print("label",train_label_1100[n])

# sentiment()


def split_er_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

    with open('er_data_1500fromall.json','r') as f:
        data = json.load(f)
    f.close()
    # isDuplicate lang   title+  body   sentiment relevance
    train_climate = {}
    count = 1782
    climate = []
    for i in tqdm(data.values()):
        # print(i['lang'],i['isDuplicate'],i['title'],i['sentiment'])
        article = i['title'] + i['body'] 
        climate.append(article)

        # tokens = tokenizer.tokenize(article) 
        # if len(tokens) > 1024:
        #     sent = nltk.sent_tokenize(article)
        #     # print(len(sent))
        #     half = int(len(sent)/2)
        #     pre_half = ""
        #     for s in sent[:half]:
        #         pre_half += s 
        #     climate.append(pre_half)
        #     post_half = ""
        #     for s in sent[half:]:
        #         post_half += s
        #     climate.append(post_half)
        # else:
        #     climate.append(article)

    print(len(climate))
    for text in climate[30:]:
        key = "train-" + str(count)
        train_climate[key] = {"text":text,"label":0}
        count+=1
        if count == 2762:
            break
    print("count final:",count)
    with open('train_climate_all.json','w') as f2:
        json.dump(train_climate,f2)
    f2.close()
    print("statistic phrase")
    # length_count(climate)

# split_er_data()