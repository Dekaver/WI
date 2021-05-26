from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import base64
import re
import json
import nltk
import time

dataframe = []
dataset = json.load(open('resource/preprocessed/data_preprocessed.json',))

#stopword english
nltk.download('punkt')
stopwords=stopwords.words('english')




for i in dataset['doc']:
    name_doc = dataset['doc'][i]
    with open(f'resource/preprocessed/{name_doc}', 'rb') as f:
        text = f.read().decode("utf-8") 
        text = text.split()
        print(text)
        text = word_tokenize(text)
        print(text)
        break
        dataframe.append(text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataframe)

X = X.T.toarray()
df = pd.DataFrame(X, index=vectorizer.get_feature_names())
print(df.head(20))

def get_similar_articles(q, df):
    start_time = time.time()
    print("query:", q)
    print("Berikut artikel dengan nilai cosine similarity tertinggi: ") 
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = {}  
    for i in range(len(df.columns)):
        value = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        if (value != 0.0):
            sim[i] = value
    end_time = time.time()

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)  
    print('result :', len(sim_sorted), '\t time :', end_time - start_time, '\n\n\n')
    for k, v in sim_sorted:
        if v != 0.0:
            print("Nilai Similaritas:", v)
            print('index dokumen', k)
            print(dataframe[k][:100:])
            print()

query = input("masukkan query ('Q' to exit) :").lower()
while(query != 'q'):
    get_similar_articles(query, df)
    query = input("\n\n\n\n\nmasukkan query ('Q' to exit) :").lower()


