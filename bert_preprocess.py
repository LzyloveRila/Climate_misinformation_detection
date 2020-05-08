import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer

def real_data_to_trainjson():
    df = pd.read_csv('./new data/all_data.csv',index_col=0)
    # print(df)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    truth_set = {}
    count = 1168
    for row in df.itertuples():
        # print(row[21])  #18 text  21 type
        if row[21] == 'real':
            key_name = 'train-' + str(count)
            print("text:",row[18])
            tokened_text = tokenizer.tokenize(row[18])
            if len(tokened_text)<1100:

                truth_set[key_name] = {"text":row[18],'label':0}
                count += 1
            if count == 4000:
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


with open('train.json','r') as f1:
    text1 = json.load(f1)
f1.close()
with open('train_1100.json','r') as f2:
    text2 = json.load(f2)
f2.close()

dataset = {**text1,**text2}
with open('train_total_1100.json','w') as f:
    json.dump(dataset,f)

# train_data = []
# # train_labels = []
# with open('train_realfact.json','r') as f:
#   data = json.load(f)
#   for v in data.values():
#     s = '[CLS] ' + v['text'] + ' [SEP]'
#     train_data.append(s)
#     # train_labels.append(v['label'])
# f.close()


def length_count():
    print("length_dataset:",len(train_data))
    # print(train_data[0])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    tokenized_train=[tokenizer.tokenize(sent) for sent in train_data] 

    length = [len(i) for i in tokenized_train]
    print(length)

    print('-------------------------')
    print("average:",np.mean(length))
    print('min:',np.min(length))
    print("max:",np.max(length))
    x_128 = 0
    x_256 = 0
    x_512 = 0
    total = 1532
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

# length_count()