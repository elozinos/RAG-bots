# 🤖 RAG Recommender Systems

Two intelligent recommendation systems built using Retrieval Augmented Generation (RAG) — a core pattern in modern AI engineering. Both systems use semantic search to find relevant content and pass it to a Large Language Model (LLM) to generate intelligent, context-aware recommendations.

---

## 📚 Projects

### 1. Book Recommender RAG
A semantic book recommendation system. Describe the kind of book you're in the mood for and the system finds and recommends the most relevant books with detailed explanations.

### 2. Movie Recommender RAG
A semantic movie recommendation system. Describe the kind of movie you want to watch and the system finds and recommends the most relevant movies complete with genre, director, and reasons why they match your interest.

---

## 🧠 How It Works

Both systems follow the same RAG pipeline:

```
INGESTION (once)
Load Dataset → Extract Metadata → Embed Text → Store in Pinecone

RETRIEVAL + GENERATION (every query)
User Input → Embed Query → Search Pinecone → Extract Docs → Build Prompt → LLM Answer
```

### The RAG Flow in Plain English
1. Your dataset is converted into numbers (vectors) that represent meaning
2. These vectors are stored in Pinecone — a vector database searchable by meaning
3. When a user asks a question it is converted into the same kind of numbers
4. Pinecone finds the most similar vectors (most relevant content)
5. The relevant content is added to a prompt alongside the user's question
6. An LLM reads the prompt and generates an intelligent recommendation
7. The answer is returned with sources cited

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenAI API | Text embeddings + LLM (gpt-4o-mini) |
| Pinecone | Vector database for semantic search |
| Pandas | Data loading and manipulation |
| NumPy | Vector/array handling |
| UUID | Generating unique record IDs |

---

## 📁 Project Structure

```
RAG-Recommender-Systems/
├── book_recommender/
│   ├── books.csv          # Book dataset
│   └── book_rag.py        # Book recommender RAG system
├── movie_recommender/
│   ├── movies.csv         # Movie dataset
│   └── movie_rag.py       # Movie recommender RAG system
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/RAG-Recommender-Systems.git
cd RAG-Recommender-Systems
```

### 2. Install dependencies
```bash
pip install openai pinecone-client pandas numpy
```

### 3. Add your API keys
In each file replace the placeholder keys:
```python
OPENAI_API_KEY = "your-openai-key"
PINECONE_API_KEY = "your-pinecone-key"
```

> ⚠️ Never share or commit your API keys publicly. Consider using environment variables or a .env file.

### 4. Run the system
```bash
# Book recommender
python book_recommender/book_rag.py

# Movie recommender
python movie_recommender/movie_rag.py
```

---

## 💬 Example Usage

### Book Recommender
```
What are you looking for?
> I want a book about magic and adventure

Based on your interest in magic and adventure, here are my recommendations:

1. Harry Potter by J.K Rowling — follows a young wizard discovering
   his powers in a magical school, perfect for magic and adventure lovers.

2. The Hobbit by J.R.R Tolkien — an unexpected adventure through
   a richly built fantasy world full of magic and danger.

Sources:
Harry Potter : J.K Rowling
The Hobbit : J.R.R Tolkien
```

### Movie Recommender
```
What are you looking for?
> I want a sci-fi movie about fighting evil forces

Based on your interest, here are my recommendations:

1. Star Wars (Sci-Fi) — follows a young farmer who discovers special
   powers and joins a rebellion against an evil empire.

2. The Matrix (Sci-Fi) — a mind bending story about fighting machines
   that control humanity, great for action and sci-fi fans.

Sources:
Star Wars : Sci-Fi : George Lucas
The Matrix : Sci-Fi : The Wachowskis
```

---

## 🔑 Key Concepts Demonstrated

- **Vector Embeddings** — converting text into numbers that represent meaning
- **Semantic Search** — searching by meaning not just keywords
- **RAG Pipeline** — retrieval augmented generation end to end
- **Pinecone Hierarchy** — Project → Index → Namespace → Records
- **Prompt Engineering** — building structured prompts with context
- **LLM Integration** — using GPT-4o-mini for intelligent generation
- **Metadata Management** — storing and retrieving human readable labels

---

## 📌 What I Learned Building This

This project was built as part of my AI engineering learning journey. Key takeaways:

- How semantic search differs from keyword search
- Why vector databases exist and how they work
- The full RAG pipeline from ingestion to generation
- How embeddings capture meaning mathematically
- How to connect multiple AI services into one pipeline

---

## 🚀 Future Improvements

- Add a web interface using Streamlit
- Expand datasets with more books and movies
- Add filtering by genre, author, or director
- Implement conversation memory for multi-turn recommendations
- Deploy as a web API using FastAPI

---

## 👤 Author

Built by Elozino as part of an AI engineering learning journey.

---

## 📄 License

MIT License — feel free to use and build on this project.
