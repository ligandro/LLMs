# RAG from Scratch - Parts 10-11: Routing and Query Construction

## Overview

Parts 10-11 introduce two critical concepts for intelligent RAG systems:

1. **Routing**: Directing queries to the most appropriate data source or prompt
2. **Query Construction**: Converting natural language into structured database queries

These techniques enable RAG systems to handle multiple data sources and leverage metadata for more precise retrieval.

---

## Part 10: Routing

### What is Routing?

**Routing** is the process of analyzing a user's question and directing it to the most appropriate:
- Data source (different databases or indexes)
- Processing chain (different prompts or models)
- Retrieval strategy (different search methods)

### Why Routing Matters

In real applications, you often have:
- Multiple specialized databases (Python docs, JavaScript docs, Go docs)
- Different types of questions (technical vs. conceptual)
- Various domain experts (physics, math, biology)

**Without routing:** All queries hit the same system, leading to:
- Irrelevant results from wrong data sources
- Generic answers when specialized knowledge is needed
- Wasted compute searching inappropriate indexes

**With routing:** Queries are intelligently directed for optimal results.

---

## Logical Routing (Function Calling)

### Concept

Use an LLM with **function calling** (structured output) to classify the question and route it to the appropriate destination.

### How It Works

The LLM acts as a "classifier" that analyzes the question and returns a structured decision about where to route it.

### Implementation

#### Step 1: Define Route Schema

```python
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )
```

**Key elements:**
- `BaseModel`: Pydantic model defining the structure
- `Literal`: Restricts choices to specific options (Python, JavaScript, or Go)
- `Field`: Adds description to guide the LLM's classification

**What this creates:**
A strict schema that forces the LLM to return one of three exact values, not free text.

#### Step 2: Create Structured LLM

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2") 
structured_llm = llm.with_structured_output(RouteQuery)
```

**`with_structured_output()`:**
- Configures the LLM to return data matching the schema
- Uses function calling under the hood
- Ensures type safety and validation

#### Step 3: Build Router Prompt

```python
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

router = prompt | structured_llm
```

**Router chain:**
1. Question → Prompt (adds system instructions)
2. Prompt → Structured LLM
3. LLM → `RouteQuery` object with selected datasource

#### Step 4: Use the Router

```python
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

result = router.invoke({"question": question})
print(result.datasource)  # Output: "python_docs"
```

**What happens:**
1. LLM analyzes the code snippet
2. Recognizes it's Python (imports from `langchain_core`)
3. Returns structured object: `RouteQuery(datasource="python_docs")`

#### Step 5: Implement Routing Logic

```python
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        return "chain for js_docs"
    else:
        return "golang_docs"

from langchain_core.runnables import RunnableLambda

full_chain = router | RunnableLambda(choose_route)
```

**Flow:**
```
Question
   ↓
Router (classifies)
   ↓
RouteQuery object
   ↓
choose_route() function
   ↓
Appropriate chain/data source
```

### Real-World Example

In practice, each route would connect to a different system:

```python
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        # Route to Python documentation retriever
        return python_rag_chain
    elif "js_docs" in result.datasource.lower():
        # Route to JavaScript documentation retriever
        return js_rag_chain
    else:
        # Route to Go documentation retriever
        return golang_rag_chain
```

Each chain would have its own:
- Vector store with domain-specific documents
- Specialized prompts
- Optimized retrieval parameters

### Benefits of Logical Routing

- ✅ **Precise**: Leverages LLM's understanding to classify accurately
- ✅ **Flexible**: Easy to add new routes by updating the Literal type
- ✅ **Type-safe**: Structured output prevents routing errors
- ✅ **Explainable**: Clear decision from the LLM

### Drawbacks

- ❌ **Requires LLM call**: Adds latency and cost
- ❌ **LLM-dependent**: Classification quality depends on model capability
- ❌ **Discrete choices**: Can only route to predefined options

---

## Semantic Routing (Embedding-Based)

### Concept

Use **embeddings and similarity search** to find the most relevant prompt or chain without an LLM call.

### How It Works

1. Pre-compute embeddings for each prompt/route
2. Embed the user's question
3. Calculate similarity between question and each prompt
4. Route to the most similar prompt

### Why This Approach?

- **Faster**: No LLM inference for routing decision
- **Cheaper**: Only embedding costs (much cheaper than LLM calls)
- **Continuous**: Similarity scores provide nuance vs. discrete classification

### Implementation

#### Step 1: Define Multiple Prompts

```python
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""
```

**Why embed whole prompts?**
- Prompts contain semantic information about their domain
- Physics prompt contains physics terminology
- Math prompt contains math terminology
- Question similarity to prompt indicates domain match

#### Step 2: Embed All Prompts

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3.2")
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)
```

**This happens once** (offline):
- Each prompt converted to a vector
- Vectors cached for fast lookup

#### Step 3: Routing Function

```python
from langchain.utils.math import cosine_similarity

def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    
    # Find most similar
    most_similar = prompt_templates[similarity.argmax()]
    
    # Chosen prompt 
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)
```

**Step-by-step:**
1. `embed_query()`: Convert question to vector
2. `cosine_similarity()`: Compare question vector to all prompt vectors
3. `argmax()`: Find index of highest similarity
4. Return corresponding prompt template

#### Step 4: Build Routing Chain

```python
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

output = chain.invoke("What's a Jacobian Matrix?")
```

**Flow:**
```
Question: "What's a Jacobian Matrix?"
   ↓
Embed question
   ↓
Compare to physics_template embedding: similarity = 0.32
Compare to math_template embedding: similarity = 0.78
   ↓
Select math_template (highest similarity)
   ↓
Fill math_template with question
   ↓
LLM generates answer using math prompt
```

### Understanding Cosine Similarity

**Cosine similarity** measures the angle between vectors:
- **1.0**: Perfect match (same direction)
- **0.0**: Orthogonal (unrelated)
- **-1.0**: Opposite (contradictory)

**Example:**
```
Question: "What's a Jacobian Matrix?"
Embedding: [0.2, 0.8, 0.1, ...]

Physics prompt embedding: [0.1, 0.3, 0.9, ...]
Math prompt embedding: [0.3, 0.7, 0.2, ...]

Cosine similarity:
- Physics: 0.32 (some overlap)
- Math: 0.78 (high overlap)

→ Route to math
```

### Benefits of Semantic Routing

- ✅ **Fast**: No LLM call for routing
- ✅ **Cheap**: Only embedding costs
- ✅ **Scalable**: Can handle many routes efficiently
- ✅ **Continuous**: Similarity scores show confidence

### Drawbacks

- ❌ **Pre-defined**: Must create prompts upfront
- ❌ **Embedding-dependent**: Quality depends on embedding model
- ❌ **Less nuanced**: May not understand complex logic like LLMs

---

## Logical vs. Semantic Routing Comparison

| Aspect | Logical (Function Calling) | Semantic (Embedding) |
|--------|---------------------------|----------------------|
| **Speed** | Slower (LLM call) | Faster (embedding only) |
| **Cost** | Higher (full LLM inference) | Lower (embedding only) |
| **Accuracy** | High (LLM reasoning) | Good (similarity-based) |
| **Flexibility** | Handles complex logic | Best for simple categorization |
| **Setup** | Define schema | Create and embed prompts |
| **Best for** | Complex routing decisions | Quick domain classification |

---

## Part 11: Query Construction (Metadata Filtering)

### Concept

Convert natural language questions into **structured queries** that can leverage metadata filters in vector databases.

### The Problem

Vector stores often contain rich metadata:

```python
Document(
    page_content="This is a tutorial about RAG...",
    metadata={
        'title': 'RAG from Scratch Tutorial',
        'view_count': 15420,
        'publish_date': '2024-01-15',
        'length': 1843,  # seconds
        'author': 'LangChain'
    }
)
```

**Without query construction:**
- Question: "Show me popular RAG videos from 2024"
- Simple search: Looks only at content similarity
- Ignores: view_count, publish_date filters

**With query construction:**
- Converts to: content_search="RAG" + min_view_count=10000 + earliest_publish_date=2024-01-01
- Much more precise retrieval!

### Why Metadata Matters

Metadata enables:
- **Temporal filtering**: "recent articles", "published in 2023"
- **Quality filtering**: "popular posts", "highly rated"
- **Categorical filtering**: "Python tutorials", "beginner level"
- **Range filtering**: "videos under 5 minutes", "articles over 1000 words"

---

## Query Construction Implementation

### Step 1: Define Search Schema

```python
import datetime
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )
```

**Schema design principles:**

1. **Semantic fields** (`content_search`, `title_search`): For similarity search
2. **Metadata filters**: Match your document metadata structure
3. **Optional fields**: Only populated when user specifies them
4. **Clear descriptions**: Guide the LLM on when/how to use each field
5. **Type safety**: Proper types (int, date, str) prevent errors

### Step 2: Create Query Analyzer

```python
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm
```

**Key instruction:** "If there are acronyms or words you are not familiar with, do not try to rephrase them."

**Why?** 
- Users often search technical terms: "RAG", "LCEL", "LangChain"
- LLM shouldn't "help" by changing "RAG" to "Retrieval Augmented Generation"
- Keep search terms as-is for better matching

### Step 3: Examples in Action

#### Example 1: Simple Content Search

```python
query_analyzer.invoke({"question": "rag from scratch"})
```

**Output:**
```
content_search: rag from scratch
title_search: rag from scratch
min_view_count: None
max_view_count: None
earliest_publish_date: None
latest_publish_date: None
min_length_sec: None
max_length_sec: None
```

**Analysis:** Simple question → only semantic search fields populated

---

#### Example 2: Temporal Filter

```python
query_analyzer.invoke({"question": "videos on chat langchain published in 2023"})
```

**Output:**
```
content_search: chat langchain
title_search: chat langchain
min_view_count: None
max_view_count: None
earliest_publish_date: 2023-01-01
latest_publish_date: 2024-01-01
min_length_sec: None
max_length_sec: None
```

**Analysis:** 
- Extracted content: "chat langchain"
- Understood "published in 2023" → date range filter
- Set appropriate date boundaries

---

#### Example 3: Complex Query with Multiple Filters

```python
query_analyzer.invoke({
    "question": "videos that are focused on the topic of chat langchain that are published before 2024"
})
```

**Output:**
```
content_search: chat langchain
title_search: chat langchain
min_view_count: None
max_view_count: None
earliest_publish_date: None
latest_publish_date: 2024-01-01
min_length_sec: None
max_length_sec: None
```

**Analysis:**
- Content focus: "chat langchain"
- "before 2024" → latest_publish_date set to 2024-01-01 (exclusive)

---

#### Example 4: Duration Filter

```python
query_analyzer.invoke({
    "question": "how to use multi-modal models in an agent, only videos under 5 minutes"
})
```

**Output:**
```
content_search: how to use multi-modal models in an agent
title_search: multi-modal models agent
min_view_count: None
max_view_count: None
earliest_publish_date: None
latest_publish_date: None
min_length_sec: None
max_length_sec: 300
```

**Analysis:**
- Content search: full question text
- Title search: condensed to key terms (as instructed)
- "under 5 minutes" → max_length_sec: 300 (5 × 60)
- LLM converted minutes to seconds!

---

## Using Structured Queries with Vector Stores

### Connecting to Vector Stores

The structured query can be used with vector stores that support metadata filtering:

```python
# Pseudocode - actual implementation varies by vector store
structured_query = query_analyzer.invoke({"question": user_question})

# Use content_search for similarity
similarity_results = vectorstore.similarity_search(
    query=structured_query.content_search,
    filter={
        "view_count": {"$gte": structured_query.min_view_count},
        "publish_date": {
            "$gte": structured_query.earliest_publish_date,
            "$lt": structured_query.latest_publish_date
        },
        "length": {"$lt": structured_query.max_length_sec}
    }
)
```

### Self-Querying Retriever

LangChain provides `SelfQueryingRetriever` that automatically:
1. Analyzes the question
2. Extracts metadata filters
3. Performs filtered similarity search

**Documentation:** https://python.langchain.com/docs/modules/data_connection/retrievers/self_query

---

## Benefits of Query Construction

### Precision

**Without metadata filtering:**
- Returns: All RAG videos regardless of date/length
- User must manually filter results
- May miss recent content buried in results

**With metadata filtering:**
- Returns: Only 2024 RAG videos under 5 minutes
- Exactly what user needs
- Higher relevance

### Efficiency

- Smaller result set → faster processing
- Less content for LLM to analyze
- Lower costs (fewer tokens)

### User Experience

Natural language is converted to precise database operations:
- "recent" → earliest_publish_date filter
- "popular" → min_view_count filter
- "short" → max_length_sec filter

Users don't need to learn query syntax!

---

## Advanced Query Construction

### Multiple Data Sources

Extend the schema to handle different content types:

```python
class MultiSourceSearch(BaseModel):
    """Search across multiple content types."""
    
    content_search: str
    content_type: Literal["video", "article", "documentation"]
    difficulty: Optional[Literal["beginner", "intermediate", "advanced"]]
    min_rating: Optional[float]
```

### Nested Filters

Support complex filter logic:

```python
class AdvancedSearch(BaseModel):
    """Advanced search with nested filters."""
    
    content_search: str
    must_have_tags: Optional[List[str]]  # All required
    should_have_tags: Optional[List[str]]  # At least one
    exclude_tags: Optional[List[str]]  # None of these
```

### Graph and SQL Query Construction

Beyond vector stores, query construction can generate:

- **Graph queries** (Cypher for Neo4j)
- **SQL queries** (for relational databases)
- **NoSQL queries** (MongoDB, etc.)

**Resources:**
- https://blog.langchain.dev/query-construction/
- https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/

---

## Routing + Query Construction Combined

These techniques work together powerfully:

```python
# Step 1: Route to appropriate data source
route = router.invoke({"question": question})

# Step 2: Construct structured query for that source
if route.datasource == "video_database":
    structured_query = video_query_analyzer.invoke({"question": question})
    results = video_retriever.invoke(structured_query)
elif route.datasource == "article_database":
    structured_query = article_query_analyzer.invoke({"question": question})
    results = article_retriever.invoke(structured_query)

# Step 3: Generate answer
answer = rag_chain.invoke({"context": results, "question": question})
```

**Benefits:**
1. Route to correct data source
2. Use domain-specific query construction
3. Leverage appropriate metadata
4. Optimal retrieval for each source

---

## Key Takeaways

### Routing

1. **Two approaches:** Logical (LLM-based) and Semantic (embedding-based)
2. **Trade-offs:** Accuracy vs. speed, cost vs. capability
3. **Use cases:** Multiple data sources, specialized domains, different strategies

### Query Construction

1. **Unlock metadata:** Use rich document metadata for precise filtering
2. **Natural language → Structure:** Convert user intent to database queries
3. **Type safety:** Structured output ensures valid queries
4. **Flexibility:** Adapt schema to your metadata structure

### When to Use

**Use Routing when:**
- Multiple distinct data sources exist
- Different questions need different handling
- Domain-specific expertise is required

**Use Query Construction when:**
- Documents have rich metadata
- Users express temporal, numerical, or categorical constraints
- Precision is more important than recall

**Combine both when:**
- Multiple specialized databases with metadata
- Complex applications with diverse content types
- Need maximum precision and efficiency

---

## Implementation Checklist

### For Routing

- [ ] Identify distinct data sources or prompts
- [ ] Choose routing strategy (logical or semantic)
- [ ] Define schema (logical) or create prompts (semantic)
- [ ] Implement routing logic
- [ ] Test with representative questions
- [ ] Monitor routing accuracy

### For Query Construction

- [ ] Analyze document metadata structure
- [ ] Design Pydantic schema matching metadata
- [ ] Write clear field descriptions
- [ ] Create query analyzer with appropriate system prompt
- [ ] Test with various filter combinations
- [ ] Integrate with vector store filtering

---

## Resources

### Routing
- [LangChain Routing Documentation](https://python.langchain.com/docs/expression_language/how_to/routing)
- [Query Analysis Routing](https://python.langchain.com/docs/use_cases/query_analysis/techniques/routing)
- [Embedding Router Cookbook](https://python.langchain.com/docs/expression_language/cookbook/embedding_router)

### Query Construction
- [Structuring Queries](https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring)
- [Self-Querying Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query)
- [Query Construction Blog](https://blog.langchain.dev/query-construction/)
- [Knowledge Graphs for RAG](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)

---

## Next Steps

With routing and query construction, you can build sophisticated RAG systems that:
- Intelligently direct queries to optimal sources
- Leverage metadata for precision filtering
- Scale to multiple domains and data types
- Provide highly relevant, targeted results

Future parts will likely cover additional advanced topics in retrieval optimization and RAG system design.
