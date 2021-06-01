from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import json
import re

dataframe = []
stopwords=stopwords.words('english')

def indexing(folder_dataset, json_dataset):
    dataset = json.load(open(folder_dataset + json_dataset,))

    for i in dataset['doc']:
        name_doc = dataset['doc'][i]
        with open(f'{folder_dataset}{name_doc}', 'rb') as f:
            text = f.read().decode("utf-8")

            text = text.lower()
            text=re.sub("</?.*?>"," <> ",text)
            text=re.sub("(\\d|\\W)+"," ",text)

            text = word_tokenize(text)

            text_without_sw = [word for word in text if not word in stopwords]
            
            text = " ".join(text_without_sw)
            
            dataframe.append(text)
            
    #menghitung term frekuensi
    cv = CountVectorizer(max_features=50000, binary=True, ngram_range=(1, 2))
    ct = cv.fit_transform(dataframe)

    norm_ct = normalize(ct, norm = "l1", axis=1)

    #menghitung documen frekuensi
    tv = TfidfVectorizer(max_features=50000, binary=True, norm=None , smooth_idf=False, ngram_range=(1, 2))
    tv.fit_transform(dataframe)

    #menghitung invers dokumen frekuensi
    tfidf_mat = norm_ct.multiply(tv.idf_).T.toarray()

    #menyimpan pada pandas
    df = pd.DataFrame(tfidf_mat, index=tv.get_feature_names())

    #mengurutkan data descending sesuai total idf terbesar
    df['rank'] = df.sum(axis=1)
    indexing = df.sort_values('rank', ascending=False)
    indexing.pop('rank')

    return indexing, indexing.index.tolist()

def get_dict_feature_name(terms):
    feature_name = {}
    feature_name['feature'] = terms
    return feature_name


def save_to_json(Data, json_filename):
    with open(json_filename, mode='w') as json_config:
        json.dump(Data, json_config)

if __name__ == "__main__":
    df, term = indexing("resource/preprocessed/","data_preprocessed.json")
    print(df.head(10))
    # print(df.index.tolist())
    from scipy import sparse
    # save sparse matrix unigram, bigram and trigram to .npz file
    sparse.save_npz("tfidf_mat.npz", sparse.csr_matrix(df))
    save_to_json( get_dict_feature_name(term),"tfidf_feature_name.json")

