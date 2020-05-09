import spacy
import pytextrank
import json
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import nltk

#pip install pytextrank
#python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

with open('train.json','r') as f:
    data = json.load(f)
f.close()

count = 0
summary_list = []
for item in tqdm(data.values()):
    text = item['text']
    # print('===============================text==========')
    # print(text)
    # print('-------------summary')
    doc = nlp(text)
    summary = ""
    for sent in doc._.textrank.summary(limit_phrases=20, limit_sentences=4):
        summary += str(sent)
        # print(sent)
    tokened_text = tokenizer.tokenize(summary)
    summary_list.append(len(summary))
    count+=1

print(np.mean(summary_list))
print(summary_list)





# examine the top-ranked phrases in the document
# for p in doc._.phrases:
#     print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
#     print(p.chunks)

