import pandas as pd
import numpy as np
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Initiate clients
client = OpenAI(api_key="your-openai-key")
pc = Pinecone(api_key="your-pinecone-key")

# Create index only if it doesn't exist
if 'rag-index' not in pc.list_indexes().names():
    pc.create_index(
        name='rag-index',
        dimension=1536,
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to index
index = pc.Index('rag-index')

# Load data
df = pd.read_csv("movies.csv")

# Extract metadata
metadata = [
    {
        "text_id": row['id'],
        "text": row['text'],
        "title": row['title'],
        "genre": row['genre'],
        "director": row['director']
    }
    for _, row in df.iterrows()    #  row not rows
]

# Extract texts, ids, embeddings
texts = df['text'].tolist()        #  directly from dataframe
ids = [str(uuid4()) for _ in range(len(texts))]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
embeds = [np.array(x.embedding) for x in response.data]

# Upsert to Pinecone
index.upsert(
    vectors=zip(ids, embeds, metadata),
    namespace="movie_recommender"
)

# Collect user input
user = input("What are you looking for? ")

user_response = client.embeddings.create(
    input=user,
    model="text-embedding-3-small"
)
user_embed = user_response.data[0].embedding

# Query Pinecone
result = index.query(
    vector=user_embed,
    include_metadata=True,
    top_k=3,
    namespace="movie_rec"
              "ommender"
)

# Extract docs and sources
docs = []
sources = []
for match in result['matches']:
    docs.append(match['metadata']['text'])          # ✅ inside metadata
    sources.append((match['metadata']['title'],     # ✅ inside metadata
                    match['metadata']['genre'],
                    match['metadata']['director']))

# Build prompt
def prompt_builder(docs, query):
    delim = "\n\n---\n\n"
    prompt_start = "Recommend a movie based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    prompt = prompt_start + delim.join(docs) + prompt_end
    return prompt

prompt = prompt_builder(docs, user)

# Send to LLM
model_response = client.chat.completions.create(
    model="gpt-4o-mini",           #  fixed model name
    messages=[
        {"role": "system", "content": "You are an expert movie recommender"},
        {"role": "user",   "content": prompt}
    ]
)

# Extract and print answer
answer = model_response.choices[0].message.content.strip()
answer += "\n\nSources:"
for source in sources:
    answer += "\n" + source[0] + " : " + source[1] + " : " + source[2]  #  all three

print(answer)