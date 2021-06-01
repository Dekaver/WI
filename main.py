from scipy import sparse
from indexing import indexing, save_to_json, get_dict_feature_name
#jalankan indexing
df, term = indexing("resource/preprocessed/","data_preprocessed.json")
print(df.head(10))

sparse.save_npz("resource/index/tfidf_mat.npz", sparse.csr_matrix(df))
save_to_json( get_dict_feature_name(term),"resource/index/tfidf_feature_name.json")

