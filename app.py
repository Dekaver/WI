# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from flask import Flask
from flask import render_template
from flask import request
from markupsafe import escape
import indexing
from ranking import ranking

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/search",methods=['GET'])
def search():
    query = request.args.get('q')  
    docs , waktu = ranking(query, indexing.df)
    return render_template('search.html', 
        docs=docs, 
        dataframe=indexing.dataframe, 
        waktu=waktu, 
        query=query,
        url=indexing.url,)

if __name__ == "__main__":
    app.run()