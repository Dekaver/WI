# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from flask import Flask
from flask import render_template
from flask import request
from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/search",methods=['GET'])
def hello():
    from scipy import sparse
    import json
    import pandas as pd

    dataset = json.load(open('resource/preprocessed/data_preprocessed.json'))
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

    indexing_data = sparse.load_npz('resource/index/tfidf_mat.npz')
    feature_name = json.load(open("resource/index/tfidf_feature_name.json"))
    feature_name = feature_name['feature']

    df = pd.DataFrame(indexing_data.toarray(), index=feature_name)

    #ranking
    from ranking import ranking
    # query = escape(name)  
    query = request.args.get('q')  
    docs , waktu = ranking(query, df)

    return render_template('test.html', docs=docs, dataframe=dataframe, waktu=waktu, query=query)

if __name__ == "__main__":
    app.run()