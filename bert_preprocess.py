import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from textblob import TextBlob
import nltk
from tqdm import tqdm
import os
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def real_data_to_trainjson():
    df = pd.read_csv('./new data/all_data.csv',index_col=0)
    # print(df)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    truth_set = {}
    count = 2100
    for row in df.itertuples():
        # print(row[21])  #18 text  21 type
        if row[21] == 'real':
            key_name = 'train-' + str(count)
            # print("text:",row[18])
            tokened_text = tokenizer.tokenize(row[18])
            if len(tokened_text)<1100 and len(tokened_text)> 25:
                truth_set[key_name] = {"text":row[18],'label':0}
                count += 1
            if count == 3400:
                break

    with open('train_fact.json','w') as f:
        json.dump(truth_set,f)

# real_data_to_trainjson()

def chechk_len():
    with open('tiny_set.json','r') as f:
        text = json.load(f)
    train_set = []
    for i,t in text.items():
        train_set.append(t['text'])
    print(len(train_set)) #1168 pos      8074 neg


def concatenate_pos_neg_set():
    with open('train_neg_new.json','r') as f1:
        text1 = json.load(f1)
    f1.close()
    with open('train_climate_test_new512.json','r') as f2:
        text2 = json.load(f2)
    f2.close()

    dataset = {**text1,**text2}
    with open('train_negitive_samples.json','w') as f:
        json.dump(dataset,f)

concatenate_pos_neg_set()




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
    print(length)


# train_data = []
# # train_labels = []
# with open('dev.json','r') as f:
#   data = json.load(f)
#   for v in data.values():
#     s = v['text'] 
#     train_data.append(s)
#     # train_labels.append(v['label'])
# f.close()
# length_count(train_data)



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
    with open('train_fact_new2.json','r') as f:
        data = f.read()
        data = json.loads(data)

    f.close()

    train_set = []
    train_label = []
    for i,t in data.items():
        train_set.append(t['text'])
        train_label.append(t['label'])

    # analyser = SentimentIntensityAnalyzer()
    # pos : compound > 0.05
    # neg : compound < -0.05
    # neu : [-0.05,0.05]

    # subjectivity 0-1
    neu,pos,neg = 0,0,0
    sub = 0
    # for sent in sent1:
    subj_list = []
    polar_list = []
    for i in tqdm(train_set):
        sentences = nltk.sent_tokenize(i)    
        doc2 = TextBlob(sentences[0])
        polarity = doc2.polarity
        subjectivity = doc2.subjectivity
        subj_list.append(round(subjectivity,2))
        polar_list.append(round(polarity,2))
        # score = analyser.polarity_scores(i)
        # score['compound]
        if subjectivity > 0.3:
            sub+=1
        if polarity > 0.0:
            pos+=1
        elif polarity < 0.0:
            neg+=1
        else:
            neu+=1
    
    print("number of pos:",pos," neg:",neg," neu:",neu)
    print(np.mean(polar_list))
    print(np.mean(subj_list))
    print('=====================')
    print(sub)

# sentiment()


def split_er_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

    with open('er_data_test_climate.json','r') as f:
        data = json.load(f)
    f.close()
    # isDuplicate lang   title+  body   sentiment relevance
    train_climate = {}
    count = 2545
    climate = []
    for i in data.values():
        # print(i['lang'],i['isDuplicate'],i['title'],i['sentiment'])
        article = i['title'] + i['body'] 
        climate.append(article)
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
    for text in tqdm(climate):
        tokens = tokenizer.tokenize(text) 
        if len(tokens) < 2000 and len(tokens) > 10:
            key = "train-" + str(count)
            train_climate[key] = {"text":text,"label":0}
            count+=1
        # if count == 3600:
        #     break
    print("count final:",count)
    with open('train_climate_test_new512.json','w') as f2:
        json.dump(train_climate,f2)
    f2.close()
    print("statistic phrase")
    length_count(climate)

# split_er_data()

def generate_polifact_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    path = './new data/politifact/real'
    # path = './new data/gossipcop/real'
    FileList = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            FileList.append(os.path.join(home, filename))
    
    print(len(FileList))
    train_fact = {}
    count = 2100
    length_list = []
    for file in tqdm(FileList):
        # print(file)
        with open(file,'r') as f:
            data = json.load(f)
            # print(data)
        f.close()
        text = data['title']+data['text']
        # print(data['keywords'])
        tokens = tokenizer.tokenize(text)
        if len(tokens) > 8 and len(tokens) < 6000:
            length_list.append(len(tokens)) 
            key_name = 'train-' + str(count)
            train_fact[key_name] = {"text":text,'label':0}
            count += 1
        if count == 3400:
            break
    print("final count:",count)
    print(np.mean(length_list))
    print("-------\n",length_list)
    with open('train_fact_new.json','w') as f1:
        json.dump(train_fact,f1)
    f1.close()
# generate_polifact_data()


def check_output():
    label1 = []
    with open('test-output1.json','r') as f:
        data=json.load(f)
        # print(data)
        for i in data.values():
            label1.append(i['label'])
    f.close()

    label2=[]
    with open('(LDA+SVM)new2test-output.json','r') as f:
        data=json.load(f)
        for i in data.values():
            label2.append(i['label'])
    f.close()

    diff = 0
    for i in range(len(label1)):
        if label1[i] != label2[i]:
            diff+=1
    print(diff)
# check_output()

