from nltk.corpus import stopwords

def ranking(query, df):
    query = query.lower()
    query=re.sub("</?.*?>"," <> ",query)
    query=re.sub("(\\d|\\W)+"," ",query)

    stopwords=stopwords.words('english')
    query = [word for word in query if not word in stopwords]
    print(query)
    start_time = time.time()
    print("query:", q)
    print("Berikut artikel dengan nilai cosine similarity tertinggi: ") 
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = {}  
    for i in range(len(df.columns)):
        value = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        if (value != 0.0):
            sim[i] = value
    end_time = time.time()

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)  
    print('result :', len(sim_sorted), '\t time :', end_time - start_time, '\n\n\n')
    for k, v in sim_sorted:
        if v != 0.0:
            print("Nilai Similaritas:", v)
            print('index dokumen', k)
            print(dataframe[k][:100:])
            print()

# query = input("masukkan query ('Q' to exit) :").lower()
# while(query != 'q'):
#     ranking(query, df)
#     query = input("\n\n\n\n\nmasukkan query ('Q' to exit) :").lower()