# RAG from Scratch - Parts 12-14: Advanced Indexing

## Overview

Parts 12-14 focus on **advanced indexing strategies** that go beyond basic chunking and embedding. These techniques optimize what gets indexed versus what gets retrieved, enabling more sophisticated and accurate RAG systems.

### Evolution of Indexing

**Basic RAG (Parts 1-4):**
```
Document → Chunk → Embed → Index chunk → Retrieve chunk
```

**Advanced Indexing (Parts 12-14):**
```
Document → Chunk → Transform/Summarize → Index transformation → Retrieve original
```

**Key insight:** What you **search** doesn't have to be what you **return**.

---

## Part 12: Multi-Representation Indexing

### Concept

**Multi-representation indexing** separates what you index from what you retrieve:
- **Index**: Summaries, keywords, or condensed representations
- **Retrieve**: Full original documents or chunks

### The Problem with Traditional Indexing

**Scenario:** A user searches for "memory in agents"

**Traditional approach:**
```
Query: "memory in agents"
   ↓
Search: Long, detailed document chunks
   ↓
Problem: Detailed chunks contain lots of specific information
   └─> May not match well with general queries
   └─> Similarity search diluted by extraneous details
```

**Example chunk:**
```
"In our implementation, we utilize a memory module that consists of 
short-term and long-term storage mechanisms. The short-term memory 
uses a sliding window approach with a capacity of 100 tokens, while 
the long-term memory employs a vector database with HNSW indexing 
for efficient retrieval. We observed that this hybrid approach 
improved performance by 23% compared to single-tier memory..."
```

**Issue:** Lots of implementation details may lower similarity with the simple query "memory in agents"

### The Solution

**Multi-representation approach:**
```
Query: "memory in agents"
   ↓
Search: Concise summary
   ↓
Match: "This document discusses memory systems in AI agents"
   ↓
Retrieve: Full original document (with all details)
```

**Benefits:**
- ✅ Search against clean, focused summaries (better matching)
- ✅ Return complete original content (no information loss)
- ✅ Best of both worlds: precision in search, completeness in retrieval

---

## Multi-Representation Implementation

### Architecture

Two separate stores:
1. **Vector Store**: Holds embeddings of summaries
2. **Document Store**: Holds original full documents

**Connection:** Each summary links to its original document via ID

### Step 1: Load Documents

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())
```

**Result:** Full documents loaded (these will be retrieved)

### Step 2: Generate Summaries

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2") 

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})
```

**What happens:**
- Each document → LLM
- LLM generates concise summary
- `batch()` with `max_concurrency=5`: Process 5 documents in parallel

**Example:**
- **Original (500 tokens)**: Long detailed discussion of agent memory systems...
- **Summary (50 tokens)**: "This article discusses memory mechanisms in AI agents, including short-term and long-term storage strategies."

### Step 3: Set Up Multi-Vector Retriever

```python
import uuid
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3.2")

# Vector store for summaries (what we search)
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embeddings
)

# Document store for full documents (what we return)
store = InMemoryByteStore()

id_key = "doc_id"

# The retriever that manages both stores
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # Searchable summaries
    byte_store=store,          # Full documents
    id_key=id_key,            # Link between them
)
```

**Architecture:**

```
┌─────────────────────────────────────┐
│      Multi-Vector Retriever         │
├─────────────────────────────────────┤
│                                     │
│  Vector Store         Doc Store    │
│  ┌─────────────┐    ┌───────────┐ │
│  │  Summary 1  │───→│  Full Doc │ │
│  │  (embedded) │ ID │     1     │ │
│  └─────────────┘    └───────────┘ │
│                                     │
│  ┌─────────────┐    ┌───────────┐ │
│  │  Summary 2  │───→│  Full Doc │ │
│  │  (embedded) │ ID │     2     │ │
│  └─────────────┘    └───────────┘ │
└─────────────────────────────────────┘
```

### Step 4: Add Documents

```python
# Generate unique IDs for each document
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Create summary documents with links to original docs
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add summaries to vector store (for searching)
retriever.vectorstore.add_documents(summary_docs)

# Add full documents to document store (for retrieval)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```

**What's stored:**

**Vector Store:**
```
Summary: "Article about AI agent memory systems..."
Metadata: {doc_id: "abc-123-def"}
```

**Document Store:**
```
ID: "abc-123-def"
Content: [Full 5000-word article about AI agent memory...]
```

### Step 5: Search and Retrieve

#### Direct Vector Store Search (summary only)

```python
query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query, k=1)
print(sub_docs[0])
```

**Returns:** Just the summary (what was indexed)

#### Multi-Vector Retriever (full document)

```python
retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
print(retrieved_docs[0].page_content[0:500])
```

**Returns:** Full original document (linked via ID)

**Process:**
1. Query embedded
2. Search summaries in vector store
3. Find most similar summary
4. Get doc_id from summary metadata
5. Retrieve full document from document store using doc_id
6. Return full document

---

## Multi-Representation Use Cases

### Use Case 1: Long Documents

**Problem:** Embedding entire research papers loses nuance

**Solution:**
- Index: Abstract or executive summary
- Retrieve: Full paper

### Use Case 2: Tables and Structured Data

**Problem:** Tables embed poorly as raw text

**Solution:**
- Index: Natural language description of table contents
- Retrieve: Original table data

**Example:**
```
Table:
| Year | Revenue | Profit |
|------|---------|--------|
| 2021 | $500M   | $50M   |
| 2022 | $750M   | $90M   |

Summary for indexing:
"Financial performance showing 50% revenue growth from 
2021 to 2022, with profit margins improving from 10% to 12%"

Retrieval: Return actual table for precise data
```

### Use Case 3: Code Documentation

**Problem:** Code snippets need context

**Solution:**
- Index: Plain English description of what code does
- Retrieve: Actual code with comments

---

## Related: Parent Document Retriever

**Similar concept** with different granularity:

- **Index**: Small chunks (better specificity)
- **Retrieve**: Larger parent chunks (more context)

**Example:**
```
Parent document: Full section (1000 tokens)
   ├── Child chunk 1 (200 tokens) ← Indexed
   ├── Child chunk 2 (200 tokens) ← Indexed
   ├── Child chunk 3 (200 tokens) ← Indexed
   └── etc.

Search matches child chunk 2
   → Retrieve entire parent section
```

**Documentation:** https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

---

## Part 13: RAPTOR

### Full Name

**RAPTOR**: Recursive Abstractive Processing for Tree-Organized Retrieval

### Concept

Build a **hierarchical tree of summaries** where:
- **Leaf level**: Original document chunks
- **Middle levels**: Summaries of clustered chunks
- **Top level**: Summary of entire document

Then search across **all levels** of the tree.

### The Problem with Flat Indexing

**Traditional RAG:**
```
Document
  ↓
Chunk 1, Chunk 2, Chunk 3, ..., Chunk 100
  ↓
All chunks at same level
  ↓
Search returns most similar chunks
```

**Issue:** Queries requiring high-level understanding may not match specific chunks

**Example:**
- **Query:** "What is the main argument of this paper?"
- **Chunks:** Contain specific examples, data, implementation details
- **Problem:** None of the chunks directly state "the main argument"

### The RAPTOR Solution

**Hierarchical organization:**

```
                    ┌─────────────────┐
                    │  Document       │
                    │  Summary        │  ← Level 2
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐    ┌───▼────┐    ┌───▼────┐
         │Cluster 1│    │Cluster2│    │Cluster3│  ← Level 1
         │ Summary │    │Summary │    │Summary │
         └────┬────┘    └───┬────┘    └───┬────┘
              │             │              │
    ┌─────┬───┴───┬─────┐  │     ┌────┬───┴───┬────┐
    │     │       │     │  │     │    │       │    │
 ┌──▼─┐┌─▼──┐ ┌──▼─┐┌──▼─┐│  ┌──▼─┐┌─▼──┐ ┌──▼─┐┌──▼─┐
 │Ch 1││Ch 2│ │Ch 3││Ch 4││  │Ch 5││Ch 6│ │Ch 7││Ch 8│  ← Level 0
 └────┘└────┘ └────┘└────┘│  └────┘└────┘ └────┘└────┘
                           │
                           │  ... more chunks
```

**Now:** Search can match at any level!

### How RAPTOR Works

#### Step 1: Chunk Documents

Split document into base chunks (leaf nodes)

#### Step 2: Cluster Chunks

Group semantically similar chunks using clustering algorithms (e.g., k-means, UMAP)

**Why cluster?**
- Related content summarized together
- Creates coherent summaries
- Preserves thematic organization

#### Step 3: Summarize Clusters

For each cluster, generate a summary

```python
Cluster = [Chunk 1, Chunk 2, Chunk 3]
   ↓
LLM summarizes all chunks in cluster
   ↓
Cluster Summary = "This section discusses X, Y, and Z..."
```

#### Step 4: Recursively Repeat

Treat cluster summaries as new chunks:
- Cluster the summaries
- Summarize those clusters
- Continue until one root summary remains

#### Step 5: Build Tree

Result is a tree where:
- **Leaves**: Original chunks (most specific)
- **Middle nodes**: Cluster summaries (medium abstraction)
- **Root**: Document summary (most abstract)

#### Step 6: Index All Nodes

**Critical:** Index ALL nodes (leaves + internal nodes) in the vector store

#### Step 7: Retrieval

When querying:
1. Search entire tree (all levels)
2. High-level query matches abstract summaries
3. Specific query matches detailed chunks
4. Return best matches regardless of level

### RAPTOR Benefits

**Multi-scale retrieval:**
- Abstract queries → Match high-level summaries
- Specific queries → Match detailed chunks
- Medium queries → Match cluster summaries

**Coherent context:**
- Clustered summaries maintain topic coherence
- Better than random similar chunks

**Hierarchical understanding:**
- Can answer both "big picture" and "detailed" questions

### Example Scenario

**Document:** 50-page research paper on neural networks

**Traditional RAG:**
- Query: "What are the main contributions?"
- Matches: Chunks mentioning "contribution" (may be scattered)
- Problem: Doesn't provide coherent overview

**RAPTOR:**
- Query: "What are the main contributions?"
- Matches: Root summary node
- Root summary: "This paper contributes three main advances: 1) Novel architecture, 2) Training technique, 3) Benchmark results"
- Perfect match!

**Another query:** "How does batch normalization work in the proposed model?"
- Matches: Specific chunk from implementation section
- Returns detailed technical explanation

### Implementation Notes

**Full implementation:** https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb

**Key steps:**
1. Clustering: UMAP for dimensionality reduction, k-means for clustering
2. Summarization: LLM generates summaries of each cluster
3. Recursion: Repeat until convergence
4. Indexing: All nodes embedded and stored
5. Retrieval: Standard similarity search across all nodes

**Video deep dive:** https://www.youtube.com/watch?v=jbGchdTL7d0

### RAPTOR vs Multi-Representation

| Aspect | Multi-Representation | RAPTOR |
|--------|---------------------|--------|
| **Structure** | Flat (summary ↔ doc pairs) | Hierarchical tree |
| **Summaries** | One per document | Multiple levels |
| **What's returned** | Original documents | Any tree node |
| **Best for** | Long documents | Multi-scale queries |
| **Complexity** | Low | High (clustering + recursion) |

### Related Paper

**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**
- Paper: https://arxiv.org/pdf/2401.18059.pdf
- Key contribution: Hierarchical indexing for multi-scale retrieval

---

## Part 14: ColBERT

### Full Name

**ColBERT**: Contextualized Late Interaction over BERT

### The Fundamental Shift

**Traditional embeddings:**
- Entire passage → Single vector
- Entire query → Single vector
- Similarity = One vector comparison

**ColBERT:**
- Each **token** in passage → Separate vector
- Each **token** in query → Separate vector
- Similarity = Multiple fine-grained comparisons

### Why This Matters

**Example:**

**Query:** "What animation studio did Miyazaki found?"

**Traditional approach:**
```
Query embedding: [0.2, 0.5, 0.1, ...]  (single vector)
Document embedding: [0.3, 0.4, 0.2, ...]  (single vector)
Similarity: cosine(query_vec, doc_vec) = 0.75
```

**ColBERT approach:**
```
Query tokens:
- "What"      → [0.1, 0.2, ...]
- "animation" → [0.5, 0.8, ...]
- "studio"    → [0.3, 0.7, ...]
- "Miyazaki"  → [0.9, 0.1, ...]
- "found"     → [0.2, 0.4, ...]

Document tokens (100+ tokens):
- "Hayao"     → [0.85, 0.15, ...]
- "Miyazaki"  → [0.88, 0.12, ...]
- "founded"   → [0.21, 0.39, ...]
- "Studio"    → [0.31, 0.69, ...]
- "Ghibli"    → [0.52, 0.79, ...]
- ... (all tokens)
```

### ColBERT Scoring Mechanism

**MaxSim operator:**

For each query token:
1. Compare to ALL document tokens
2. Take the MAXIMUM similarity
3. Sum these maximum similarities

**Formula:**
```
Score = Σ max_sim(q_token, all_doc_tokens) for each q_token
```

**Step-by-step example:**

Query token: "Miyazaki"
├─ vs doc token "Hayao":    similarity = 0.45
├─ vs doc token "Miyazaki": similarity = 0.99 ← MAX
├─ vs doc token "founded":  similarity = 0.12
├─ vs doc token "Studio":   similarity = 0.23
└─ vs doc token "Ghibli":   similarity = 0.31
→ Max similarity for "Miyazaki" = 0.99

Query token: "studio"
├─ vs doc token "Hayao":    similarity = 0.15
├─ vs doc token "Miyazaki": similarity = 0.20
├─ vs doc token "founded":  similarity = 0.18
├─ vs doc token "Studio":   similarity = 0.95 ← MAX
└─ vs doc token "Ghibli":   similarity = 0.42
→ Max similarity for "studio" = 0.95

... repeat for all query tokens ...

Final score = 0.99 + 0.95 + 0.88 + ... (sum of all max similarities)
```

### Why Token-Level Embeddings Work Better

**Advantage 1: Precision**
- Can match specific important terms
- "Miyazaki" query token can find exact matches
- Entire-passage embedding might dilute this signal

**Advantage 2: Context-aware**
- Each token embedding considers surrounding context
- "bank" near "river" vs "bank" near "money" get different embeddings
- Better disambiguation

**Advantage 3: Compositional**
- Multi-word concepts captured at token level
- "Studio Ghibli" = strong matches on both tokens
- More robust than hoping phrase appears in training

**Advantage 4: Partial Matches**
- Even if full phrase doesn't match, important tokens do
- Query: "Miyazaki's animation company"
- Matches: "Miyazaki", "animation", "company" tokens individually
- Works even if exact phrase never appears

### ColBERT vs Traditional Embeddings

| Aspect | Traditional | ColBERT |
|--------|-------------|---------|
| **Granularity** | Passage-level | Token-level |
| **Vectors per doc** | 1 | Hundreds (one per token) |
| **Storage** | Low | Higher |
| **Precision** | Moderate | High |
| **Context** | Passage context | Token context |
| **Speed** | Fast (single comparison) | Slower (many comparisons) |

---

## ColBERT Implementation with RAGatouille

### What is RAGatouille?

A library that makes ColBERT easy to use, providing:
- Simple API
- Pre-trained models
- Integration with LangChain
- Efficient indexing and search

### Step 1: Install

```python
! pip install -U ragatouille
```

### Step 2: Load Pre-trained Model

```python
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
```

**Model:** ColBERTv2.0 - state-of-the-art pre-trained model

### Step 3: Get Documents

```python
import requests

def get_wikipedia_page(title: str):
    """Retrieve the full text content of a Wikipedia page."""
    URL = "https://en.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}
    
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")
```

**Result:** Full Wikipedia article text

### Step 4: Index Documents

```python
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)
```

**Parameters:**
- `collection`: List of documents to index
- `index_name`: Name for this index (for later retrieval)
- `max_document_length`: Maximum tokens per chunk
- `split_documents=True`: Automatically chunk long documents

**What happens:**
1. Document split into chunks (max 180 tokens each)
2. Each chunk tokenized
3. Each token embedded with context
4. All token embeddings stored in index

### Step 5: Search

```python
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
print(results)
```

**Returns:** Top 3 most relevant chunks with scores

**Search process:**
1. Query tokenized and embedded
2. MaxSim scoring against all indexed chunks
3. Chunks ranked by score
4. Top k returned

### Step 6: LangChain Integration

```python
retriever = RAG.as_langchain_retriever(k=3)
docs = retriever.invoke("What animation studio did Miyazaki found?")
```

**Use in RAG chain:**
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What animation studio did Miyazaki found?")
```

**Integration:** ColBERT retriever works seamlessly in LangChain pipelines

---

## ColBERT Deep Dive

### Contextualized Embeddings

**Key innovation:** Token embeddings are contextualized

**What this means:**
```
Sentence 1: "I went to the bank to deposit money"
Sentence 2: "I sat by the river bank"

Traditional word embedding:
- "bank" → Same vector in both sentences

ColBERT:
- "bank" in sentence 1 → [0.7, 0.2, 0.1, ...] (financial context)
- "bank" in sentence 2 → [0.2, 0.8, 0.3, ...] (geographical context)
```

**How:** BERT-based model considers surrounding tokens

### Late Interaction

**"Late" means:** Interaction happens at query time, not indexing time

**Traditional (early interaction):**
```
Index time: Document → Single embedding (interaction complete)
Query time: Compare embeddings (just distance calculation)
```

**ColBERT (late interaction):**
```
Index time: Document → Token embeddings (no interaction yet)
Query time: Query tokens interact with document tokens (MaxSim)
```

**Benefit:** More flexible, query-specific scoring

### Performance Characteristics

**Speed:**
- Slower than traditional single-vector search
- Much faster than re-ranking every document with BERT
- Good balance between accuracy and speed

**Storage:**
- Requires more space (vector per token vs per document)
- Compression techniques available
- Trade-off: accuracy vs storage

**Accuracy:**
- Significantly better than traditional embeddings
- Approaches re-ranking quality
- Especially good for precise queries

---

## When to Use Each Technique

### Multi-Representation Indexing

**Use when:**
- ✅ Documents are long and detailed
- ✅ You want to preserve full context
- ✅ Summaries can improve search precision
- ✅ Simple implementation is preferred

**Don't use when:**
- ❌ Documents are already concise
- ❌ Summary generation is expensive
- ❌ You need real-time indexing

### RAPTOR

**Use when:**
- ✅ Need to answer both high-level and detailed questions
- ✅ Documents have hierarchical structure
- ✅ Users ask questions at different abstraction levels
- ✅ Document collections are large

**Don't use when:**
- ❌ All queries are at same level of detail
- ❌ Documents are short
- ❌ Computation budget is limited
- ❌ Simple flat structure suffices

### ColBERT

**Use when:**
- ✅ Precision is critical
- ✅ Queries contain specific terms to match
- ✅ Domain contains ambiguous terms needing context
- ✅ Standard embeddings underperform

**Don't use when:**
- ❌ Storage is constrained
- ❌ Latency must be minimal
- ❌ Standard embeddings work well enough
- ❌ Very large-scale systems (billions of docs)

---

## Combining Techniques

These approaches can be combined!

### Multi-Representation + ColBERT

```
1. Generate summaries of documents
2. Index summaries with ColBERT (token-level precision)
3. Retrieve linked full documents
```

**Benefit:** Precision of ColBERT + completeness of full docs

### RAPTOR + ColBERT

```
1. Build RAPTOR tree
2. Index all tree nodes with ColBERT
3. MaxSim scoring across hierarchical levels
```

**Benefit:** Multi-scale retrieval + token-level precision

---

## Advanced Indexing Checklist

### Planning

- [ ] Analyze query patterns (abstract vs specific)
- [ ] Evaluate document characteristics (length, structure)
- [ ] Determine precision requirements
- [ ] Assess computational budget

### Multi-Representation

- [ ] Identify what to index (summaries, descriptions)
- [ ] Choose summarization strategy
- [ ] Set up dual stores (vector + document)
- [ ] Implement ID linking
- [ ] Test retrieval quality

### RAPTOR

- [ ] Choose clustering algorithm
- [ ] Determine tree depth
- [ ] Configure summarization prompts
- [ ] Build hierarchical index
- [ ] Evaluate multi-scale retrieval

### ColBERT

- [ ] Install RAGatouille
- [ ] Load pre-trained model
- [ ] Configure chunking parameters
- [ ] Index documents
- [ ] Integrate with RAG pipeline

---

## Key Takeaways

### The Indexing Mindset

**Traditional thinking:** "How should I chunk my documents?"

**Advanced thinking:** 
- "What should I index vs what should I retrieve?"
- "At what granularity should I match queries?"
- "How can I support different query types?"

### The Paradigm Shift

**From:** Document → Chunk → Embed → Index → Retrieve

**To:** Document → Transform (summary/hierarchy/tokens) → Index → Match → Retrieve (original)

### Performance Trade-offs

All advanced techniques trade computational cost for accuracy:
- Multi-Representation: Summary generation cost
- RAPTOR: Clustering + hierarchical summarization cost
- ColBERT: Storage + token-level comparison cost

**Choose based on:** Use case requirements, budget, scale

---

## Resources

### Multi-Representation Indexing
- [Semi-Structured Multi-Modal RAG](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- [MultiVectorRetriever Docs](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
- [Parent Document Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
- Paper: https://arxiv.org/abs/2312.06648

### RAPTOR
- [Full Implementation](https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb)
- [Video Deep Dive](https://www.youtube.com/watch?v=jbGchdTL7d0)
- Paper: https://arxiv.org/pdf/2401.18059.pdf

### ColBERT
- [RAGatouille Integration](https://python.langchain.com/docs/integrations/retrievers/ragatouille)
- [ColBERT Overview](https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag)
- [Simon Willison's Notes](https://til.simonwillison.net/llms/colbert-ragatouille)

### Document Chunking
- [Greg Kamradt's Video on Chunking](https://www.youtube.com/watch?v=8OJC21T2SL4)

---

## Next Steps

With advanced indexing techniques, you can:
- Build multi-scale retrieval systems
- Support diverse query types
- Achieve higher precision and recall
- Optimize for your specific use case

These techniques represent the cutting edge of RAG indexing and form the foundation for production-quality systems handling complex retrieval scenarios.
