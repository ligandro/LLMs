# 📚 The Great Gatsby - Vector Database Project

A complete implementation of a vector database using ChromaDB to make F. Scott Fitzgerald's "The Great Gatsby" semantically searchable.

## 🎯 What This Project Does

This project demonstrates how to:
- Download and preprocess classic literature from Project Gutenberg
- Create vector embeddings using SentenceTransformers
- Store embeddings in ChromaDB for efficient semantic search
- Perform semantic queries to find relevant passages

## ✨ Features

- **📖 Text Processing**: Automatically downloads and cleans The Great Gatsby text
- **🔢 Smart Chunking**: Splits the novel into meaningful paragraphs for better search results
- **🧠 Vector Embeddings**: Uses the `all-MiniLM-L6-v2` model (384 dimensions)
- **💾 Persistent Storage**: Stores the vector database locally for reuse
- **🔍 Semantic Search**: Find passages by meaning, not just keywords
- **📊 Interactive Queries**: Multiple example queries with detailed results

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install chromadb sentence-transformers requests
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

Open and run `gatsby_vector_db.ipynb` in Jupyter:

```bash
jupyter notebook gatsby_vector_db.ipynb
```

### 3. Query the Database

Try searching for themes and topics:
- "Gatsby's extravagant parties"
- "the green light symbolism"
- "Daisy and Tom's relationship"
- "the valley of ashes"
- "eyes of Doctor T.J. Eckleburg"

## 📁 Project Structure

```
GatsbyVectorDB/
├── gatsby_vector_db.ipynb    # Main notebook
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── data/                      # Downloaded text files
│   └── great_gatsby.txt      # (Auto-downloaded)
└── gatsby_vector_db/         # ChromaDB storage
    └── ...                    # (Auto-created)
```

## 🔬 How It Works

1. **Download**: Fetches The Great Gatsby from Project Gutenberg
2. **Clean**: Removes headers, footers, and excess whitespace
3. **Chunk**: Splits into paragraphs (~300+ chunks)
4. **Embed**: Creates 384-dimensional vectors for each chunk
5. **Store**: Saves to ChromaDB with persistent storage
6. **Query**: Performs semantic search using cosine similarity

## 💡 Example Queries

### Using Manual Embeddings
```python
query = "Gatsby's parties"
query_embedding = sentence_transformer_ef([query])

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)
```

### Using Automatic Embeddings
```python
results = collection.query(
    query_texts=["Gatsby's mysterious wealth"],
    n_results=3
)
```

## 📊 Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Vector Dimensions**: 384
- **Database**: ChromaDB with persistent storage
- **Similarity Metric**: Cosine similarity
- **Text Source**: Project Gutenberg (Public Domain)

## 🎓 Learning Outcomes

After completing this project, you'll understand:
- Vector embeddings and semantic search
- ChromaDB operations and persistence
- Text preprocessing and chunking strategies
- Similarity-based retrieval systems
- Working with SentenceTransformers

## 📝 Notes

- First run downloads The Great Gatsby (~300KB)
- Embedding generation takes 1-2 minutes
- Database is saved locally for future use
- All text is public domain

## 🔗 Related Topics

- **Topics/Vector_Databases**: ChromaDB tutorials and concepts
- **Projects/RAG-Document-Chat**: RAG implementation using vector databases

---

**Built with**: ChromaDB • SentenceTransformers • Python • Jupyter
