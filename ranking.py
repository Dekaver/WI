from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import re

stopwords=stopwords.words('english')

def ranking(query, df):
    query = re.sub("</?.*?>"," <> ",query)
    query = re.sub("(\\d|\\W)+"," ",query)
    query = word_tokenize(query)
    query = [word for word in query if not word in stopwords]

    #membuat query menjadi beberapa unigram dan bigram
    keywords = []
    j = ''
    for i in query:
        keywords.append(i)
        if j != '':
            keywords.append(f'{j} {i}')
        j = i
    keys = []

    #hapus unigram "buah" jika terdapat index bigram "buah mangga"
    for key in keywords: 
        if key in df.index.tolist():
            keys.append(key)
            words = key.split(' ')
            if len(words) >1:
                for word in words:
                    if word in keys:
                        keys.remove(word)

    start_time = time.time()

    rank_doc = df.loc[keys]
    rank_doc.loc['ranking_value',:] = rank_doc.sum(axis=0)
    relavan_doc = rank_doc.loc[:, (rank_doc != 0).any(axis=0)]
    relavan_doc = relavan_doc.loc['ranking_value']
    relavan_doc = relavan_doc.sort_values(ascending=False)
    list_doc = relavan_doc.index.tolist()
    
    end_time = time.time()

    return list_doc, end_time - start_time
