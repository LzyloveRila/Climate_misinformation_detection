from eventregistry import *
KEY = "d567b42a-f8bd-4023-94c2-b8b530de5568"
er = EventRegistry(apiKey = KEY)

# q = QueryArticlesIter(conceptUri = er.getConceptUri("Environment"),lang=["eng"],
#         sourceUri=QueryItems.OR([er.getSourceUri("apple"),er.getSourceUri("bbc"),er.getSourceUri("cnn"),er.getSourceUri("nytimes")]))
q = QueryArticlesIter(keywords=QueryItems.OR(['climate change','global warming']),lang=["eng"],
        sourceUri=er.getSourceUri("news"))
data = {}
count = 0

for art in q.execQuery(er, sortBy = "rel"):
    print(art)
    key = "article-" + str(count)
    count+=1
    data[key] = art
    if count == 500:
        break
print(count)
with open('er_data_apple_climate.json','w') as f:
    json.dump(data,f)
f.close()