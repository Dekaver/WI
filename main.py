from scipy import sparse
import json
import pandas as pd

dataset = json.load(open('resource/preprocessed/data_preprocessed.json'))
url = json.load(open('resource/dataset.json'))['url']
dataframe = []
for i in dataset['doc']:
    name_doc = dataset['doc'][i]
    with open(f'resource/preprocessed/{name_doc}', 'rb') as f:
        text = f.read().decode("utf-8")
        
        text = " ".join(" ".join(str(text).split()).split('\\n'))
        dataframe.append(text)


def re_index():
    from indexing import indexing, save_to_json, get_dict_feature_name
    #jalankan indexing
    df, term = indexing("resource/preprocessed/","data_preprocessed.json")
    sparse.save_npz("resource/index/tfidf_mat.npz", sparse.csr_matrix(df))
    save_to_json( get_dict_feature_name(term),"resource/index/tfidf_feature_name.json")

# re_index()

indexing_data = sparse.load_npz('resource/index/tfidf_mat.npz')
feature_name = json.load(open("resource/index/tfidf_feature_name.json"))
feature_name = feature_name['feature']

df = pd.DataFrame(indexing_data.toarray(), index=feature_name)
index = df.to_dict('index')

print(df.head(20))
# print(feature_name)