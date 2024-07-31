import pandas as pd
import json
import ollama
import chromadb


belge = input("belge: ")
belge = "belge"+str(belge)

with open(f'surya/{belge}/results.json', 'r', encoding="utf-8-sig") as f:
    data = json.load(f)


data = data[belge][0]["text_lines"]


df = pd.DataFrame(columns=["text","bbox","c-x","c-y"])
for d in data:
    df.loc[len(df)] = d["text"],d["bbox"],d["bbox"][0]+d["bbox"][2],d["bbox"][1]+d["bbox"][3]

df["c-x"] = df["c-x"]/2
df["c-y"] = df["c-y"]/2

last_line = float('-inf')
lines =[]
for i in range (len(df)):
    #her bir kelime için
    l_i = df.loc[i]
    if(l_i["c-y"]-last_line >4):
        last_line = l_i["c-y"]
        arr=pd.DataFrame(columns=["text","c-x"])
        arr.loc[len(arr)]= l_i["text"],l_i["c-x"]
        for j in range(i+1,len(df)):
            #diğer kelimeleri
            l_j = df.loc[j]
            #aynı satırdaysa    
            if (abs(l_i["c-y"]-l_j["c-y"]) < 4):
                arr.loc[len(arr)]= l_j["text"],l_j["c-x"]
                
    
        df_sorted = arr.sort_values(by='c-x')
        t = ""
        for value in df_sorted['text']:
            t+=" "+value
        lines.append(t)


lines = [text.lower() for text in lines]

df['x-start'] = df['bbox'].apply(lambda x: x[0])
df_sorted = df.sort_values(by='x-start')
dif = float("-inf")
df_sorted = df.sort_values(by='x-start').reset_index(drop=True)
columns = []
for i in range (len(df_sorted)):
    l_i = df_sorted.loc[i]
    #farklı bir sütunsa
    if(l_i["x-start"]-dif > 15):
        dif = l_i["x-start"]
        arr = pd.DataFrame(columns=["text","x-start","c-y"])
        arr.loc[len(arr)] = l_i["text"],l_i["x-start"],l_i["c-y"]

        for j in range(len(df_sorted)):
            if (i!=j):
                #diğer kelimeleri
                l_j = df.loc[j]
                #aynı sütundalarsa  
                if (abs(l_i["x-start"]-l_j["x-start"]) < 15):
                    arr.loc[len(arr)]= l_j["text"],l_j["x-start"],l_j["c-y"]
                    
    
        arr = arr.sort_values(by='c-y').reset_index(drop=True)

        t = ""

        for value in arr['text']:
            t+=" "+value
        columns.append(t)

        

for i in range(len(columns)):
    columns[i] = str(i)+".sütun ,"+ str(columns[i])

columns = [text.lower() for text in columns]



client = chromadb.Client()


collection = client.get_or_create_collection(name="lines")
for i, d in enumerate(lines):
    response = ollama.embeddings(model="nomic-embed-text", prompt=d)
    embedding = response["embedding"]
    collection.add(ids=[str(i)],embeddings=[embedding],documents=[d])


collection = client.get_or_create_collection(name="columns")
for i, d in enumerate(columns):
    response = ollama.embeddings(model="nomic-embed-text", prompt=d)
    embedding = response["embedding"]
    collection.add(ids=[str(i)],embeddings=[embedding],documents=[d])




def send_request(req):
    prompt = req

    collection = client.get_or_create_collection(name="lines")
    response = ollama.embeddings(prompt=prompt,model="nomic-embed-text")
    results = collection.query(query_embeddings=[response["embedding"]],n_results=10)
    data_lines = results['documents'][0]
    
    collection = client.get_or_create_collection(name="columns")
    response = ollama.embeddings(prompt=prompt,model="nomic-embed-text")
    results = collection.query(query_embeddings=[response["embedding"]],n_results=10)
    data_columns = results['documents'][0]
    
    
    output = ollama.generate(model="gemma2",prompt=f"{prompt}, bu soruyu cevaplayabilmek için önce Bu bilgileri incele: {data_lines}, eğer yeterli bir cevap bulamazsan bunu incele: {data_columns} , EN mantıklı! ve EN kısa! cevabı ver")
    return (output['response'])

while True:
   req = input("soru: ")
   req = req.lower()
   print(send_request(req))
