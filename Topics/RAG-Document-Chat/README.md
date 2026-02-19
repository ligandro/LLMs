# 📚 RAG Document Chat 

A Streamlit-based AI-powered document assistant for PDF querying using LangChain, Ollama's Llama3.2, and ChromaDB.

<p align="center">
  <img width="70%" src="demo.png"> &nbsp &nbsp
</p>

---

## ⚡ Quick Demo

Upload any PDF → Ask questions → Get accurate, grounded answers

**What it does:**
- Automatically processes and chunks your PDF documents
- Embeds content into a searchable vector database
- Retrieves diverse, relevant information using MMR algorithm
- Generates answers grounded solely in document context
- Prevents hallucinations with anti-hallucination prompting

---

## 🎯 Key Features

✅ **No Hallucinations** - Answers only from document content  
✅ **Intelligent Retrieval** - MMR algorithm for diverse, relevant results  
✅ **Fast Processing** - Efficient PDF chunking and embedding  
✅ **Multi-Query Support** - Alternative retrieval method available  
✅ **Session Management** - Auto-cleanup between different PDFs  
✅ **Interactive UI** - Streamlit interface with sidebar controls  
✅ **Local LLM** - Runs entirely on Ollama (privacy-first)  
✅ **Configurable** - Easy-to-modify settings in config.py  

---

## 🏗️ How It Works

### Three Simple Stages

**Stage 1: Document Processing**
```
PDF Upload
    ↓
Extract Text
    ↓
Smart Chunking (1200 chars, 300 overlap)
    ↓
Vector Embeddings
```

**Stage 2: Intelligent Retrieval**
```
User Question
    ↓
Find Similar Context (MMR)
    ↓
Return Top 12 Diverse Chunks
```

**Stage 3: Grounded Answer**
```
Context + Question
    ↓
LLaMA 3.2 (temperature=0)
    ↓
Grounded Answer (no hallucinations)
```

### Why MMR Instead of Similarity?

**Basic Approach:** Pure semantic similarity  
→ Returns redundant chunks from same document section

**MMR Approach:** Maximal Marginal Relevance  
→ Balances relevance + diversity for comprehensive context (fetches 60, selects top 12)

---

## 📦 Installation

### Requirements
- Python 3.10+
- Ollama installed and running
- 8GB RAM minimum

### Setup

**1. Install Ollama**
```bash
# Download from https://ollama.ai
# Pull required models:
ollama pull llama3.2
ollama pull nomic-embed-text
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

**Start the application**
```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

**Usage:**
1. Upload a PDF file
2. Wait for processing (chunking + embedding)
3. Ask questions in the chat interface
4. Switch retrieval methods in sidebar (MMR/Multi-Query)

**That's it!** The system automatically cleans up when you upload a new PDF.

---

## 📚 Project Structure

```
RAG-Document-Chat/
├── ingest/                    # Document processing pipeline
│   ├── load_pdf.py           # PDF loading & cleanup
│   ├── chunk_documents.py    # Smart chunking
│   └── embed_chunks.py       # Embedding & ChromaDB
├── rag/                       # RAG components
│   ├── retriever.py          # MMR & Multi-Query retrievers
│   └── chain.py              # LLM chain with anti-hallucination
├── config.py                  # Centralized configuration
├── app.py                     # Streamlit UI
└── README.md                  # This file
```

---

## 🙏 Acknowledgments

- **LLM:** [Ollama](https://ollama.ai/) - Local LLM runtime
- **Embeddings:** [Nomic Embed Text](https://ollama.ai/library/nomic-embed-text)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/)
- **Framework:** [LangChain](https://www.langchain.com/)
- **UI:** [Streamlit](https://streamlit.io/)
- **Inspiration:** [EpsteinFiles-RAG](https://github.com/AnkitNayak-eth/EpsteinFiles-RAG) - MMR retrieval & anti-hallucination patterns

---
