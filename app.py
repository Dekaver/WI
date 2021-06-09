# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from flask import Flask
from flask import render_template
from flask import request
from markupsafe import escape
import main

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/search",methods=['GET'])
def search():
    #ranking
    from ranking import ranking
    query = request.args.get('q')  
    docs , waktu = ranking(query, main.df)
    
    return render_template('search.html', 
        docs=docs, 
        dataframe=main.dataframe, 
        waktu=waktu, 
        query=query,
        url=main.url,)

if __name__ == "__main__":
    app.run()