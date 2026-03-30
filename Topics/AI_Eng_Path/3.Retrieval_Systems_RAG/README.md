# Step 3 (4–5 Weeks): Retrieval Systems (RAG)

## Outcome
Build reliable retrieval pipelines that ground LLM answers in external knowledge.

## Topics to Cover
- Embeddings and vector databases
- Chunking and indexing strategies
- Semantic search and ranking
- End-to-end RAG pipelines
- Handling large knowledge bases

## Weekly Plan
### Week 1
- Learn embedding concepts and similarity search.
- Compare embedding models for quality vs speed.
- Store vectors in Chroma (or equivalent).

### Week 2
- Implement multiple chunking strategies:
  - Fixed-size chunking
  - Semantic chunking
  - Overlap tuning
- Evaluate retrieval quality impacts.

### Week 3
- Build retrieval + generation chain.
- Add metadata filters, top-k, and score thresholds.
- Implement citation-style output (source chunks).

### Week 4
- Handle larger corpora with incremental indexing.
- Add caching and batch ingestion.
- Introduce reranking for better relevance.

### Week 5 
- RAG Evaluation: create a benchmark set of queries with known answers.

## Hands-On Projects
1. **Single-Document QA RAG**
   - Ingest one long PDF/text source.
   - Build answer generation with citations.
   - Tune chunking and top-k for best accuracy.

2. **Multi-Document Knowledge Assistant**
   - Ingest a folder of docs.
   - Filter by metadata (author/date/topic).
   - Return sourced answers with confidence hints.

## Evaluation Metrics
- Retrieval hit rate@k
- Faithfulness to sources
- Answer completeness
- Latency (ingest and query)

## Completion Criteria
- You can ingest, chunk, embed, index, and query a document corpus.
- You can tune chunking/retrieval parameters based on measured quality.
- You can produce grounded answers with source transparency.

## Suggested Deliverable
Create a GitHub repo: **`rag-from-scratch`** with ingestion scripts, retrieval evaluation, and a simple app interface.
