# RAG from Scratch - Parts 1-4: Concepts Explained

## Overview

This notebook demonstrates the fundamental concepts of **RAG (Retrieval Augmented Generation)**, a technique that enhances Large Language Model (LLM) responses by providing them with relevant context retrieved from external documents.

RAG addresses a key limitation of LLMs: they can only work with information they were trained on. RAG allows LLMs to access and utilize up-to-date or domain-specific information by retrieving relevant documents before generating a response.

---

## The RAG Pipeline

The RAG process consists of three main stages:

### 1. **Indexing** (Offline)
### 2. **Retrieval** (At Query Time)
### 3. **Generation** (At Query Time)

---

## Part 1: Indexing Pipeline

Indexing is the process of preparing your documents so they can be efficiently searched and retrieved later. This is typically done once, offline.

### Step 1.1: Document Loading

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(file_path=doc_path)
docs = loader.load()
```

**What's happening:**
- Documents are loaded from external sources (in this case, a PDF file)
- The loader converts the PDF into a structured format that can be processed
- Each document contains text content and metadata

### Step 1.2: Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

**Why split documents?**
- Documents are often too large to process as a single unit
- Smaller chunks make retrieval more precise
- Chunks are easier to embed and compare

**Parameters:**
- `chunk_size=1000`: Each chunk will be approximately 1000 characters
- `chunk_overlap=200`: Adjacent chunks share 200 characters to maintain context at boundaries

**How RecursiveCharacterTextSplitter works:**
- Attempts to split on natural boundaries (paragraphs, sentences, words)
- Uses a hierarchy of separators: `\n\n`, `\n`, ` `, `""`
- Recursively tries each separator until chunks are small enough

### Step 1.3: Embedding

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3.2")
```

**What are embeddings?**
- Numerical vector representations of text
- Similar texts have similar vector representations
- Enable semantic search (meaning-based, not just keyword matching)

**Example:**
- "What is a dog?" and "Tell me about canines" will have similar embeddings
- Despite different words, the semantic meaning is captured

### Step 1.4: Vector Store

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)
```

**What is a vector store?**
- A specialized database for storing and searching embeddings
- Enables fast similarity search over high-dimensional vectors
- Chroma is one popular vector database; others include Pinecone, Weaviate, FAISS

**What happens during indexing:**
1. Each text chunk is converted to an embedding vector
2. Vectors are stored in the database with their original text
3. An index is built for efficient similarity search

---

## Part 3: Retrieval

Retrieval is the process of finding the most relevant documents from the vector store based on a user's query.

### Creating a Retriever

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**Parameters:**
- `k=3`: Retrieve the top 3 most relevant documents

### How Retrieval Works

```python
docs = retriever.get_relevant_documents("What is a LLM?")
```

**Process:**
1. The query "What is a LLM?" is embedded using the same embedding model
2. The vector store finds the k documents with the most similar embeddings
3. Similarity is typically measured using cosine similarity or Euclidean distance
4. The original text chunks are returned

**Why this works:**
- Semantic search finds documents by meaning, not just keywords
- The question "What is a LLM?" will match chunks that discuss LLMs, even if they don't contain those exact words

---

## Part 4: Generation

Generation is the final step where an LLM uses the retrieved context to generate an informed answer.

### The Prompt Template

```python
from langchain.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
```

**Why use a template?**
- Provides clear instructions to the LLM
- Ensures the LLM stays grounded in the provided context
- "based only on the following context" helps prevent hallucination

**Template variables:**
- `{context}`: Filled with the retrieved documents
- `{question}`: Filled with the user's question

### The LLM

```python
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")
```

**What is an LLM?**
- Large Language Model - a neural network trained on vast amounts of text
- Can understand and generate human-like text
- Examples: GPT-4, Claude, Llama, etc.

### Building the RAG Chain

The notebook demonstrates two approaches to building a RAG chain:

#### Simple Chain

```python
chain = prompt | llm
chain.invoke({"context": docs, "question": "What is LLM?"})
```

**Components:**
- `|` is the pipe operator in LangChain Expression Language (LCEL)
- Data flows from left to right: prompt → llm
- Prompt is filled with context and question, then passed to LLM

#### Full RAG Chain

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
```

**How this works:**
1. **Input**: A question string (e.g., "What is Task Decomposition?")
2. **Retrieval**: The question is passed to the retriever, which finds relevant docs
3. **Format**: `format_docs` combines multiple docs into a single string
4. **Prompt**: Context and question are inserted into the template
5. **LLM**: The LLM generates an answer based on the prompt
6. **Parse**: `StrOutputParser` extracts the text response

**RunnablePassthrough:**
- Passes the input through unchanged
- Here, it passes the question string to the prompt's `{question}` variable

---

## Key Concepts Summary

### RAG vs. Standard LLM

| Standard LLM | RAG |
|--------------|-----|
| Limited to training data | Can access external knowledge |
| May hallucinate facts | Grounded in retrieved context |
| Static knowledge cutoff | Can use up-to-date information |
| General responses | Domain-specific, accurate responses |

### The Three Pillars

1. **Indexing**: Prepare documents for efficient retrieval
   - Load → Split → Embed → Store

2. **Retrieval**: Find relevant information
   - Convert query to embedding → Search vector store → Return top-k matches

3. **Generation**: Create informed responses
   - Insert context into prompt → LLM generates answer → Parse output

### LangChain Components Used

- **Document Loaders**: Load data from various sources (PDFs, web, databases)
- **Text Splitters**: Break large documents into manageable chunks
- **Embeddings**: Convert text to vector representations
- **Vector Stores**: Store and search embeddings efficiently
- **Retrievers**: Interface for querying vector stores
- **Prompts**: Structure LLM inputs
- **LLMs**: Generate responses
- **Chains**: Combine components into pipelines using LCEL

### Advanced Features Mentioned

- **Hub.pull**: Access pre-built prompts from LangChain Hub
  ```python
  prompt = hub.pull("rlm/rag-prompt")
  ```

- **LCEL (LangChain Expression Language)**: Composable syntax for building chains
  - Pipe operator `|` chains components
  - Dictionary syntax `{"key": component}` creates parallel branches
  - Enables streaming, async, and batching automatically

---

## When to Use RAG

RAG is ideal for:
- **Knowledge bases**: Company wikis, documentation, research papers
- **Current information**: News, stock prices, recent events
- **Private data**: Information not in the LLM's training set
- **Fact-checking**: Grounding LLM responses in verifiable sources
- **Long documents**: Books, manuals, legal documents

RAG is not needed for:
- General knowledge questions the LLM already knows
- Creative writing without factual constraints
- Tasks that don't require external information

---

## Next Steps

The concepts in Parts 1-4 provide the foundation for RAG. Future parts will likely cover:
- Advanced retrieval techniques
- Query transformation and routing
- Multi-query retrieval
- Re-ranking and filtering
- Conversational RAG with memory
- Evaluation and optimization

---

## Resources

- [LangChain RAG Quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
- [LangChain Expression Language](https://python.langchain.com/docs/expression_language/get_started)
- [RAG Search Example](https://python.langchain.com/docs/expression_language/get_started#rag-search-example)
