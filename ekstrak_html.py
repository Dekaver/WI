from urllib.request import urlopen
import http.client as http
from bs4 import BeautifulSoup
# from nltk.corpus import stopwords
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
 
import json

f = open('resource/dataset.json',)

data = json.load(f)
data_preproccess = {}
data_preproccess['doc'] = {}
len(data)
for i in data['url']:
    url = data['url'][i]
    try:
        html = urlopen(url).read()
    except (http.IncompleteRead) as e:
        html = e.partial
    soup = BeautifulSoup(html, features="html5lib")

    for script in soup(["script", "style"]):
        script.extract()   
        
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    number = 1000 + int(i)
    data_preproccess['doc'][i] = f'document_{str(number)[1::]}.txt'
    
    with open(f'resource/preprocessed/document_{str(number)[1::]}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'write success document_{str(number)[1::]}.txt')
with open('resource/preprocessed/data_preprocessed.json', 'w') as f:
    json.dump(data_preproccess, f)