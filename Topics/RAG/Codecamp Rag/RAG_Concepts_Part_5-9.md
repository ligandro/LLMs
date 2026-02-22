# RAG from Scratch - Parts 5-9: Query Transformations

## Overview

While Parts 1-4 introduced the basic RAG pipeline, Parts 5-9 focus on **Query Transformations** - advanced techniques for improving retrieval quality by re-writing or modifying user questions before searching.

### The Problem

Simple similarity search has limitations:
- **Narrow scope**: A single query may not capture all relevant information
- **Poor wording**: User questions may not match how information is phrased in documents
- **Complexity**: Complex questions may be too broad or too specific
- **Vocabulary mismatch**: Query terms may differ from document terms

### The Solution: Query Transformation

Transform the original question into one or more optimized queries that:
- Capture multiple perspectives
- Use better terminology
- Break down complexity
- Bridge vocabulary gaps

---

## Part 5: Multi Query

### Concept

Instead of using a single query, generate **multiple versions** of the same question from different perspectives, then retrieve documents for each version.

### Why It Works

- Different phrasings may match different relevant documents
- Overcomes limitations of distance-based similarity search
- Increases recall (finding more relevant documents)

### Implementation

#### Step 1: Generate Multiple Query Versions

```python
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)
```

**What happens:**
- LLM receives the original question
- Generates 5 alternative phrasings
- Each version approaches the question from a different angle

**Example:**
- **Original**: "What is task decomposition for LLM agents?"
- **Variations**:
  - "How do LLM agents break down complex tasks?"
  - "What methods do language model agents use to decompose problems?"
  - "Explain the process of dividing tasks in LLM-based systems"
  - "What is the concept of task breakdown in AI agents?"
  - "How is task decomposition implemented in large language model agents?"

#### Step 2: Parse Multiple Queries

```python
generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Pipeline:**
1. Prompt filled with question
2. LLM generates multiple questions (separated by newlines)
3. Parse to string
4. Split by newlines to get list of questions

#### Step 3: Retrieve for Each Query

```python
retrieval_chain = generate_queries | retriever.map() | get_unique_union
```

**Key components:**
- `retriever.map()`: Apply retriever to each question in the list
- `get_unique_union()`: Combine results and remove duplicates

#### Step 4: Get Unique Documents

```python
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]
```

**Process:**
1. Flatten the list of document lists
2. Serialize each document to string (for comparison)
3. Use `set()` to get unique documents
4. Deserialize back to document objects

### Benefits

- ✅ Broader coverage of relevant documents
- ✅ Reduces risk of missing information due to poor query wording
- ✅ Simple to implement

### Drawbacks

- ❌ More API calls (generates queries + retrieves for each)
- ❌ No ranking - all retrieved docs treated equally

---

## Part 6: RAG-Fusion

### Concept

Similar to Multi Query, but with **Reciprocal Rank Fusion (RRF)** - a sophisticated method for combining and re-ranking results from multiple queries.

### Key Difference from Multi Query

| Multi Query | RAG-Fusion |
|-------------|------------|
| Union of all results | Weighted fusion based on ranks |
| No ranking | RRF scoring re-ranks results |
| Equal treatment | Higher-ranked docs score better |

### Implementation

#### Step 1: Generate Related Queries

```python
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
```

**Note:** Generates *related* queries, not just different phrasings

**Example:**
- **Original**: "What is task decomposition for LLM agents?"
- **Related queries**:
  - "LLM agent task planning strategies"
  - "How AI agents organize complex workflows"
  - "Task breakdown techniques in autonomous agents"
  - "Hierarchical task structures for language models"

#### Step 2: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort by fused score (descending)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results
```

### Understanding RRF Formula

**Formula:** `score += 1 / (rank + k)`

**Parameters:**
- `rank`: Position in the retrieved list (0 for first, 1 for second, etc.)
- `k`: Constant (typically 60) to prevent division by zero and smooth scores

**How it works:**

Imagine a document appears in results from 3 different queries:
- Query 1: Position 0 (1st) → score += 1/(0+60) = 0.0167
- Query 2: Position 2 (3rd) → score += 1/(2+60) = 0.0161
- Query 3: Position 10 (11th) → score += 1/(10+60) = 0.0143
- **Total score**: 0.0471

A document only appearing once:
- Query 1: Position 0 (1st) → score = 0.0167

**Result:** Documents appearing in multiple result sets, especially at high positions, get higher scores.

### Benefits

- ✅ Better ranking than simple union
- ✅ Documents confirmed by multiple queries rank higher
- ✅ Reduces impact of individual query weaknesses
- ✅ No training required (unsupervised method)

### Drawbacks

- ❌ More complex than Multi Query
- ❌ Still requires multiple retrievals

---

## Part 7: Decomposition

### Concept

Break down **complex questions** into simpler **sub-questions** that can be answered independently, then combine the answers.

### Why It Works

- Complex questions often require multiple pieces of information
- Sub-questions can be more focused and retrieve better results
- Mirrors how humans approach complex problems

### Example

**Complex question:**
"What are the main components of an LLM-powered autonomous agent system?"

**Decomposed sub-questions:**
1. "What is the architecture of LLM-powered agents?"
2. "What are the key modules in autonomous agent systems?"
3. "How do LLMs integrate with agent frameworks?"

### Two Approaches

The notebook demonstrates two strategies for handling decomposed questions:

---

### Approach 1: Answer Recursively

![Recursive approach - each answer builds on previous ones]

#### How It Works

1. Generate sub-questions
2. Answer them **sequentially**
3. Each answer becomes context for the next question
4. Final answer uses all previous Q&A pairs

#### Implementation

```python
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
```

**Key insight:** The prompt includes `{q_a_pairs}` - previously answered questions and answers

#### Process

```python
q_a_pairs = ""
for q in questions:
    rag_chain = (
        {"context": itemgetter("question") | retriever, 
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
```

**Flow:**
1. Q1 → retrieve docs → answer with no prior context → save Q1+A1
2. Q2 → retrieve docs → answer with Q1+A1 as context → save Q2+A2
3. Q3 → retrieve docs → answer with Q1+A1 and Q2+A2 as context → save Q3+A3
4. Continue building context chain...

#### Benefits

- ✅ Later answers can use information from earlier answers
- ✅ Maintains coherence across sub-questions
- ✅ Good for questions with dependencies

#### Drawbacks

- ❌ Sequential processing (slower)
- ❌ Errors cascade (wrong early answer affects later ones)
- ❌ Token usage grows with each iteration

---

### Approach 2: Answer Individually (Parallel)

![Parallel approach - all sub-questions answered independently]

#### How It Works

1. Generate sub-questions
2. Answer them **independently** and in parallel
3. Collect all answers
4. Synthesize final answer from all Q&A pairs

#### Implementation

```python
def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """RAG on each sub-question"""
    
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    
    rag_results = []
    
    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({
            "context": retrieved_docs, 
            "question": sub_question
        })
        rag_results.append(answer)
    
    return rag_results, sub_questions
```

**Note:** Each sub-question is answered independently

#### Final Synthesis

```python
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)
```

**Process:**
1. All sub-questions answered independently
2. Format all Q&A pairs together
3. LLM synthesizes comprehensive answer from all pairs

#### Benefits

- ✅ Faster (can parallelize)
- ✅ No error cascading
- ✅ Each sub-question gets focused retrieval

#### Drawbacks

- ❌ Sub-questions don't benefit from each other
- ❌ May have redundancy in answers
- ❌ Requires synthesis step

### Comparison

| Aspect | Recursive | Individual |
|--------|-----------|------------|
| **Speed** | Slower (sequential) | Faster (parallel) |
| **Context** | Cumulative | Independent |
| **Errors** | Cascade | Isolated |
| **Best for** | Dependent sub-questions | Independent sub-questions |

### Related Papers

- **Least-to-Most Prompting**: https://arxiv.org/pdf/2205.10625.pdf
- **IRCoT (Interleaving Retrieval with Chain-of-Thought)**: https://arxiv.org/abs/2212.10509.pdf

---

## Part 8: Step Back Prompting

### Concept

Instead of retrieving based on the specific question, generate a more **generic, high-level "step back" question** first, retrieve based on both, then answer.

### Why It Works

- Specific questions may be too narrow
- Generic questions retrieve broader context and principles
- Combination provides both specific details and general understanding

### Example

**Specific question:**
"Could the members of The Police perform lawful arrests?"

**Step-back question:**
"What can the members of The Police do?"

The step-back question retrieves general information about the band "The Police", which provides context to understand they're musicians, not law enforcement.

### Implementation

#### Few-Shot Prompting

```python
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?",
    },
]
```

**Purpose:** Shows the LLM what "stepping back" means through examples

#### Prompt Structure

```python
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
    ),
    # Few shot examples
    few_shot_prompt,
    # New question
    ("user", "{question}"),
])
```

**Few-shot learning:** The model learns the pattern from examples

#### Dual Retrieval

```python
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)
```

**Key insight:** Retrieves twice:
1. Using the original specific question → `normal_context`
2. Using the step-back question → `step_back_context`

Both contexts are provided to the LLM for answering.

### Response Prompt

```python
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
```

**Strategy:** LLM has both:
- Specific context (for details)
- General context (for principles and background)

### Benefits

- ✅ Combines specificity with broader understanding
- ✅ Helps with questions requiring background knowledge
- ✅ Reduces over-fitting to specific terminology

### Drawbacks

- ❌ Requires well-designed few-shot examples
- ❌ Double retrieval increases latency
- ❌ May retrieve irrelevant general information

### Related Paper

- **Take a Step Back**: https://arxiv.org/pdf/2310.06117.pdf

---

## Part 9: HyDE (Hypothetical Document Embeddings)

### Concept

Instead of searching with the **question**, generate a **hypothetical answer** (as if it were from a document), then search using that hypothetical answer's embedding.

### The Insight

**Problem:** Questions and answers are semantically different types of text
- Questions contain interrogative language: "What is...", "How does..."
- Documents contain declarative language: "X is...", "Y works by..."
- Their embeddings may not match well

**Solution:** Generate a hypothetical document passage that would answer the question, then search using that.

### Why It Works

- Hypothetical answer is stylistically similar to real documents
- Better semantic alignment in embedding space
- Bridges the question-document gap

### Example

**Question:**
"What is task decomposition for LLM agents?"

**Hypothetical passage (generated by LLM):**
"Task decomposition in LLM agents refers to the process of breaking down complex tasks into smaller, manageable sub-tasks. This approach enables autonomous agents to handle sophisticated problems by dividing them into sequential steps. Common methods include Chain-of-Thought prompting and hierarchical planning..."

**Then:** Search using the embedding of this hypothetical passage, not the original question.

### Implementation

#### Step 1: Generate Hypothetical Document

```python
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
)
```

**Note:** `temperature=0` for consistent, factual generation

#### Step 2: Retrieve Using Hypothetical Document

```python
retrieval_chain = generate_docs_for_retrieval | retriever 
retrieved_docs = retrieval_chain.invoke({"question": question})
```

**What happens:**
1. Question → LLM generates hypothetical passage
2. Hypothetical passage → embedded
3. Embedding → used to search vector store
4. Retrieve documents similar to the hypothetical passage

#### Step 3: Answer with Retrieved Documents

```python
final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context": retrieved_docs, "question": question})
```

**Important:** The final answer uses:
- The **original question** (not the hypothetical passage)
- The **retrieved documents** (found using the hypothetical passage)

### Visual Flow

```
Question
   ↓
Generate hypothetical document
   ↓
Embed hypothetical document
   ↓
Search vector store
   ↓
Retrieve real documents
   ↓
Answer original question with real documents
```

### Benefits

- ✅ Better semantic alignment between query and documents
- ✅ Works well for technical and scientific domains
- ✅ Can improve retrieval quality significantly

### Drawbacks

- ❌ Adds LLM generation step (latency + cost)
- ❌ Hypothetical document may be inaccurate
- ❌ Works best when LLM has some domain knowledge

### Related Paper

- **Precise Zero-Shot Dense Retrieval (HyDE)**: https://arxiv.org/abs/2212.10496

---

## Comparison of All Techniques

| Technique | Key Idea | Queries Generated | Ranking Method | Best For |
|-----------|----------|-------------------|----------------|----------|
| **Multi Query** | Different perspectives | 5 variations | Union (no ranking) | Broad coverage |
| **RAG-Fusion** | Related queries + fusion | 4 related queries | Reciprocal Rank Fusion | Balanced ranking |
| **Decomposition** | Break into sub-questions | 3-5 sub-questions | Synthesis | Complex questions |
| **Step Back** | Generic + specific | 1 generic version | Both contexts | Conceptual questions |
| **HyDE** | Hypothetical answer | 1 hypothetical doc | Standard similarity | Domain-specific queries |

---

## When to Use Each Technique

### Multi Query
- User question may be poorly worded
- Want maximum recall (find all relevant info)
- Less concerned about precision

### RAG-Fusion
- Need balanced results from multiple perspectives
- Want to leverage ranking signals
- Looking for consensus information

### Decomposition (Recursive)
- Complex question with dependencies
- Each part builds on previous answers
- Coherent narrative needed

### Decomposition (Individual)
- Complex question with independent parts
- Can parallelize for speed
- Want isolated error handling

### Step Back
- Question requires background knowledge
- Specific question may be too narrow
- Want both specific and general context

### HyDE
- Domain-specific or technical questions
- Traditional keyword search fails
- LLM has reasonable domain knowledge

---

## Combining Techniques

These techniques can be combined for even better results:

**Example combinations:**
1. **Step Back + RAG-Fusion**: Generate step-back question, then create multiple versions of both questions
2. **Decomposition + Multi Query**: Decompose the question, then generate multiple versions of each sub-question
3. **HyDE + RAG-Fusion**: Generate hypothetical document, extract key concepts, create related queries

---

## Implementation Patterns

### Common Chain Structure

Most techniques follow this pattern:

```python
# 1. Query transformation
transformed_queries = transform_chain.invoke({"question": question})

# 2. Retrieval
docs = retriever_chain.invoke(transformed_queries)

# 3. Generation
answer = rag_chain.invoke({"context": docs, "question": question})
```

### LangChain Expression Language (LCEL) Utilities

**Parallel branches:**
```python
{
    "context": retriever,
    "question": RunnablePassthrough()
}
```

**Mapping over lists:**
```python
retriever.map()  # Apply retriever to each item in a list
```

**Lambda functions:**
```python
RunnableLambda(lambda x: x.split("\n"))
```

**Item extraction:**
```python
itemgetter("question")  # Extract specific key from dict
```

---

## Key Takeaways

1. **One size doesn't fit all**: Different questions benefit from different transformations

2. **Trade-offs exist**: More sophisticated techniques cost more in latency and API calls

3. **Experimentation is key**: Test techniques on your specific domain and use cases

4. **Quality over quantity**: Better transformations > more transformations

5. **Context matters**: Understanding your documents and user questions guides technique selection

6. **Combine wisely**: Stacking too many techniques can add complexity without proportional benefit

---

## Resources

### Papers
- **Least-to-Most Prompting**: https://arxiv.org/pdf/2205.10625.pdf
- **IRCoT**: https://arxiv.org/abs/2212.10509.pdf
- **Step-Back Prompting**: https://arxiv.org/pdf/2310.06117.pdf
- **HyDE**: https://arxiv.org/abs/2212.10496

### Code & Documentation
- [LangChain MultiQueryRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
- [RAG-Fusion Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb)
- [HyDE Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb)

### Blog Posts
- [Forget RAG, the Future is RAG-Fusion](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)
