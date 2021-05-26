import urllib
import http.client as http
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
 
import json
import re

f = open('resource/dataset.json',)
stopwords=stopwords.words('english')

def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    return text

data = json.load(f)
data_preproccess = {}
data_preproccess['doc'] = {}
for i in data['url']:
    if int(i) <= 71 :
        continue
    url = data['url'][i]
    try:
        html = urllib.request.urlopen(url).read()
    except (http.IncompleteRead) as e:
        html = e.partial
    soup = BeautifulSoup(html, features="html5lib")

    for script in soup(["script", "style"]):
        script.extract()   
        
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text.replace('\n', ' ')
    text = pre_process(text)
    text = ' '.join(' '.join(str(text).split()).split('\\n'))
        # text = word_tokenize(text)
        # dataframe.append(pre_process(text))
    text = pre_process(text).split()
    tokens_without_sw = [word for word in text if not word in stopwords]
    doc = ' '.join(tokens_without_sw)


    number = 1000 + int(i)
    data_preproccess['doc'][i] = f'document_{str(number)[1::]}.txt'
    
    with open(f'resource/preprocessed/document_{str(number)[1::]}.txt', 'w') as f:
        f.write(doc)
        print(f'write success document_{str(number)[1::]}.txt')
# with open('resource/preprocessed/data_preprocessed.json', 'w') as f:
#     json.dump(data_preproccess, f)