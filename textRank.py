"""
This scirpt uses the textRank library to extract the summary from articles
"""

import spacy
import pytextrank
import json
from tqdm import tqdm
import numpy as np
import nltk

#pip install pytextrank
#python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

# extract the key sentences in the articles as bert do not support long text
with open('test-unlabelled.json','r') as f:
    data = json.load(f)
f.close()

def main():
    number_of_sentences = []
    train_total_extracted = {}
    for item in tqdm(data.items()):
        text = item[1]['text']
        # statistic the article length by num of sentences
        sent_token = nltk.sent_tokenize(text)
        number_of_sentences.append(len(sent_token))

        # print('===============================text==========================')
        # print(text)
        # print('-------------summary')
        doc = nlp(text)
        summary = ""
        for sent in doc._.textrank.summary(limit_phrases=20, limit_sentences=4):
            summary += str(sent)

        train_total_extracted[item[0]] = {'text':summary,'label':item[1]['label']}   
    
    with open('dev_extracted.json','w') as f2:
        json.dump(train_total_extracted,f2)
    f2.close()
    print(np.mean(number_of_sentences))
    print("===========================================================")
    print(number_of_sentences)


# main()

def test_extract():
    number_of_sentences = []
    train_total_extracted = {}
    for item in tqdm(data.items()):
        text = item[1]['text']
        # statistic the article length by num of sentences
        sent_token = nltk.sent_tokenize(text)
        number_of_sentences.append(len(sent_token))

        # print('===============================text==========================')
        # print(text)
        # print('-------------summary')
        doc = nlp(text)
        summary = ""
        for sent in doc._.textrank.summary(limit_phrases=20, limit_sentences=4):
            summary += str(sent)

        train_total_extracted[item[0]] = {'text':summary}   
    
    with open('test_extracted.json','w') as f2:
        json.dump(train_total_extracted,f2)
    f2.close()
    print(np.mean(number_of_sentences))
    print("===========================================================")
    print(number_of_sentences)

test_extract()