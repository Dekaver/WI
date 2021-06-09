
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import json
import re
import pytest
import requests

dataframe = []

def test_render_main_page():
    url = "http://127.0.0.1:5000/"
    req = requests.get(url)
    assert req.status_code == 200, "render success"

def test_load_dataset():
    dataset = json.load(open("resource/preprocessed/data_preprocessed.json"))
    assert dataset
    
def test_download_stopword_list():
    import nltk
    assert nltk.download('stopwords')
    assert nltk.download('punkt')

def test_make_dataframe_from_dataset():
    stopwords_list=stopwords.words('english')
    folder_dataset = "resource/preprocessed/"
    nama_file = "data_preprocessed.json"
    dataset = json.load(open(folder_dataset+nama_file))

    for i in dataset['doc']:
        name_doc = dataset['doc'][i]
        with open(f'{folder_dataset}{name_doc}', 'rb') as f:
            text = f.read().decode("utf-8")

            text = text.lower()
            text=re.sub("</?.*?>"," <> ",text)
            text=re.sub("(\\d|\\W)+"," ",text)

            text = word_tokenize(text)

            text_without_sw = [word for word in text if not word in stopwords_list]
            
            text = " ".join(text_without_sw)
            
            dataframe.append(text)

    assert dataframe

# integration
def test_search_without_query():
    url = "http://127.0.0.1:5000/search?q="
    req = requests.get(url)
    assert req.status_code == 200, "search success"

def test_search_with_query():
    url = "http://127.0.0.1:5000/search?q="
    query = "index out of range"
    req = requests.get(url+query)
    assert req.status_code == 200, "search success"