from scipy import sparse
import scipy
import json
import pandas as pd

def re_index():
    from indexing import indexing, save_to_json, get_dict_feature_name
    #jalankan indexing
    df, term = indexing("resource/preprocessed/","data_preprocessed.json")
    sparse.save_npz("resource/index/tfidf_mat.npz", sparse.csr_matrix(df))
    save_to_json( get_dict_feature_name(term),"resource/index/tfidf_feature_name.json")

re_index()

indexing_data = scipy.sparse.load_npz('resource/index/tfidf_mat.npz')
feature_name = json.load(open("resource/index/tfidf_feature_name.json"))
feature_name = feature_name['feature']

df = pd.DataFrame(indexing_data.toarray(), index=feature_name)
df['rank'] = df.sum(axis=1)
print(df.head(10))
# print(indexing_data.toarray())

