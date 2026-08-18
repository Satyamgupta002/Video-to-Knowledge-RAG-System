# to convert little chunks to vector embeddings

import requests
import os
import json
# import numpy as np
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

jsons = os.listdir("E:\RAG Based AI Course Assistant\merged_jsons")
my_dicts = [] #each and every chunk of all json files along with their embedding will come here
chunk_id = 0
for json_file in jsons:  # for every json file
    with open(f"E:\RAG Based AI Course Assistant\merged_jsons\{json_file}") as f:
        content = json.load(f)
    print(f"Converting chunks to embeddings: {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    for i, chunk in enumerate(content['chunks']): #for every chunk of this file
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
    
df = pd.DataFrame.from_records(my_dicts)
print(df.shape)
joblib.dump(df, 'embeddings2.joblib')