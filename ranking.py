from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import re

stopwords=stopwords.words('english')

def ranking(query, df):
    start_time = time.time()
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
    print(keywords)

    #hapus unigram "buah" jika terdapat index bigram "buah mangga"
    for key in keywords:
        if key in df.index.tolist():
            keys.append(key)
            words = key.split(' ')
            if len(words) >1:
                for word in words:
                    if word in keys:
                        keys.remove(word)


    rank_doc = df.loc[keys]
    rank_doc.loc['ranking_value',:] = rank_doc.sum(axis=0)
    relavan_doc = rank_doc.loc[:, (rank_doc != 0).any(axis=0)]
    relavan_doc = relavan_doc.loc['ranking_value']
    relavan_doc = relavan_doc.sort_values(ascending=False)
    list_doc = relavan_doc.index.tolist()
    end_time = time.time()

    return list_doc, end_time - start_time

    # print("Berikut artikel dengan nilai cosine similarity tertinggi: ") 
    # q = [q]
    # q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    # sim = {}  
    # for i in range(len(df.columns)):
    #     value = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    #     if (value != 0.0):
    #         sim[i] = value
    # end_time = time.time()

    # sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)  
    # print('result :', len(sim_sorted), '\t time :', end_time - start_time, '\n\n\n')
    # for k, v in sim_sorted:
    #     if v != 0.0:
    #         print("Nilai Similaritas:", v)
    #         print('index dokumen', k)
    #         print(dataframe[k][:100:])
    #         print()

if __name__ == "__main__":
    import scipy.sparse
    import json
    import pandas as pd
    indexing_data = scipy.sparse.load_npz('resource/index/tfidf_mat.npz')
    feature_name = json.load(open("resource/index/tfidf_feature_name.json"))
    feature_name = feature_name['feature']

    df = pd.DataFrame(indexing_data.toarray(), index=feature_name)
    # print(df.head(10))
    query = input("masukkan query ('Q' to exit) :").lower()
    while(query != 'q'):
        doc , waktu = ranking(query, df)
        print(waktu)
        print(doc)
        query = input("\n\n\n\n\nmasukkan query ('Q' to exit) :").lower()