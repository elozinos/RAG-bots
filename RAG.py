from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from uuid import uuid4
import numpy as np

# Load CSV once
df = pd.read_csv('books.csv')

# API keys — store these safely, never share them!
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_5B2fF9_3n5HnM6jTav1EmMweJCayEtk1V43nyaXanKLobPJtQ9YgB2nCueidgJfqGTwX2A"

# Initiate clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index only if it doesn't exist
if 'my-index' not in pc.list_indexes().names():
    pc.create_index(
        name='my-index',
        dimension=1536,
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to index
index = pc.Index('my-index')

# Extract metadata
metadatas = [
    {
        "text_id": row['id'],
        "text": row['text'],
        "title": row['title'],
        "author": row['author']
    }
    for _, row in df.iterrows()
]

# Extract texts, ids, embeddings
texts = df['text'].tolist()
ids = [str(uuid4()) for _ in range(len(texts))]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeds = [np.array(x.embedding) for x in response.data]

# Push to Pinecone
index.upsert(
    vectors=zip(ids, embeds, metadatas),
    namespace="book_recommender"
)

# Collect user input and embed it
user = input("What are you looking for? ")

user_response = client.embeddings.create(
    input=user,
    model="text-embedding-3-small"
)
user_embedd = user_response.data[0].embedding

# Query Pinecone and print results
results = index.query(
    vector=user_embedd,
    top_k=3,
    namespace="book_recommender",
    include_metadata=True
)

#extract docs and sources


def docs_extract(results, query):
    docs=[]
    sources=[]
    for matches in results['matches']:
        docs.append(matches['metadata']['text'])
        sources.append((matches['metadata']['author'], matches['metadata']['title']))
    return docs, sources


def prompt_builder(docs, query):
    delim = "\n\n---\n\n"
    prompt_start = "Recommend a book based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"

    prompt = prompt_start + delim.join(docs) + prompt_end

    return prompt

docs, sources= docs_extract(results, user)
prompt=prompt_builder(docs, user)
model_response=client.chat.completions.create(

    messages=[{"role":"system","content":"you're an expert book recommeder"},
              {"role":"user","content": prompt}],
    model="gpt-40-mini"
)

answer = model_response.choices[0].message.content.strip()
answer += "\n\nSources:"
for source in sources:
    answer += "\n" + source[0] + " : " + source[1]
print(answer)

# PINECONE QUERY RESPONSE HIERARCHY:
#
# results  (dictionary)
# ├── 'matches'  (list of matched books)
# │       ├── match 1  (dictionary)
# │       │       ├── 'id'        → the uuid you generated
# │       │       ├── 'score'     → how similar to query (1.0 = perfect match)
# │       │       └── 'metadata'  (dictionary)
# │       │               ├── 'text_id' → original id from CSV
# │       │               ├── 'text'    → the book summary
# │       │               ├── 'title'   → the book title
# │       │               └── 'author'  → the author name
# │       ├── match 2  (dictionary)
# │       │       └── same structure as match 1...
# │       └── match 3  (dictionary)
# │               └── same structure as match 1...
# └── 'namespace' → where the data came from