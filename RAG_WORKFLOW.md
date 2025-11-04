# RAG System Workflow and Document Retrieval Techniques

## Overview

This document provides a step-by-step explanation of the workflow and techniques employed in building the Retrieval-Augmented Generation (RAG) system, with a focus on document relevance classification and retrieval. The system uses a two-stage retrieval approach: initial semantic search via vector embeddings, followed by optional cross-encoder reranking for improved relevance scoring.

## System Architecture

The RAG pipeline consists of several interconnected stages:

1. **Document Ingestion & Processing**
2. **Embedding Generation**
3. **Vector Store Indexing**
4. **Query Processing & Retrieval**
5. **Relevance Reranking (Classification)**
6. **Context-Aware Response Generation**

---

## Step-by-Step Workflow

### Phase 1: Document Ingestion and Preprocessing

#### 1.1 Document Loading
- **Location**: `loaders/loader.py`
- **Process**:
  - Scans the `data/` directory for `.txt` files
  - Reads UTF-8 encoded text content
  - Skips empty files and handles encoding errors gracefully
  - Creates LangChain `Document` objects with raw text content

#### 1.2 Metadata Extraction
- **Location**: `loaders/metadata.py`
- **Techniques**:
  - **Publication ID Extraction**: Extracts document identifier from filename (stem without extension)
  - **Topic Classification**: Uses keyword-based matching against predefined topic buckets:
    - 5G, Security, Partnership, Product, Sustainability
  - **Date Extraction**: Regex pattern matching for dates in formats `DD.MM.YYYY` or `YYYY-MM-DD`
  - **Company Extraction**: Heuristic-based identification using capitalized bigram patterns
  - **Word Count**: Simple whitespace-based tokenization
  - **Content Hashing**: SHA-256 hash for deduplication

**Metadata Structure**:
```python
{
    "publication_id": "doc_001",
    "source": "Deutsche Telekom",
    "extracted_at": "2024-01-15T10:30:00Z",
    "word_count": 1250,
    "mentioned_dates": ["2022-01-15", "15.01.2023"],
    "topics": ["5G", "Security"],
    "mentioned_companies": ["Microsoft Corporation", "SAP SE"]
}
```

#### 1.3 Text Chunking
- **Location**: `core/chunking.py`
- **Strategy**: Metadata-Aware Chunking with RecursiveCharacterTextSplitter
- **Parameters**:
  - `chunk_size`: 500 characters (default)
  - `chunk_overlap`: 100 characters (default)
  - Separators: `["\n\n", "\n", ". ", " ", ""]` (hierarchical splitting)

**Chunking Process**:
1. Splits text using hierarchical separators (paragraphs → sentences → words)
2. Preserves document boundaries and metadata
3. Generates stable chunk IDs using SHA-256 hash of `{doc_id}_{chunk_index}_{first_50_chars}`
4. Attaches metadata to each chunk:
   - Source document ID
   - Chunk index and total chunk count
   - All original document metadata (topics, dates, companies)

**Chunk Metadata Preservation**:
```python
{
    "source": "Deutsche Telekom",
    "doc_id": "doc_001",
    "chunk_index": 0,
    "total_chunks": 5,
    "chunk_id": "a1b2c3d4e5f6g7h8",
    "publication_id": "doc_001",
    "topics": ["5G", "Security"],
    # ... all other document metadata
}
```

---

### Phase 2: Embedding Generation and Vectorization

#### 2.1 Embedding Model Selection
- **Location**: `core/embeddings.py`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384-dimensional vectors
- **Rationale**: 
  - Lightweight and efficient
  - Good balance between quality and speed
  - Compatible with LangChain's embedding interface

#### 2.2 Embedding Generation
- **Process**:
  - Each chunk is passed through the SentenceTransformer model
  - Generates dense vector representations capturing semantic meaning
  - Vectors are normalized and stored for similarity computation

**Embedding Properties**:
- **Semantic Similarity**: Captures meaning, not just keyword matching
- **Dimensionality**: 384 dimensions per chunk
- **Normalization**: Cosine similarity for retrieval

---

### Phase 3: Vector Store Indexing

#### 3.1 ChromaDB Integration
- **Location**: `core/retrieval.py`, `scripts/ingest.py`
- **Vector Database**: ChromaDB (persistent, local-first)
- **Indexing Strategy**:
  - Documents are stored with their embeddings
  - Metadata is indexed for filtering
  - Persistence to `.chroma/` directory

#### 3.2 Index Structure
- **Collection**: Single collection containing all document chunks
- **Stored Information**:
  - Embedding vectors (384-dim)
  - Document metadata (publication_id, topics, dates, etc.)
  - Chunk identifiers and indices
  - Source text (for retrieval)

---

### Phase 4: Query Processing and Initial Retrieval

#### 4.1 Query Embedding
- **Process**:
  1. User query is received via API or UI
  2. Same embedding model encodes query into 384-dimensional vector
  3. Query vector is normalized for cosine similarity computation

#### 4.2 Vector Similarity Search
- **Location**: `core/retrieval.py` → `AdvancedRetriever.retrieve()`
- **Algorithm**: Cosine similarity search in ChromaDB
- **Process**:
  1. ChromaDB computes cosine similarity between query embedding and all chunk embeddings
  2. Returns top-k candidates (default k=5, configurable up to 20)
  3. Results are sorted by similarity score (descending)

**Initial Retrieval Configuration**:
- **top_k**: Number of final results (default: 5)
- **rerank_top_k**: Number of candidates for reranking (default: 10, if reranking enabled)
- **Metadata Filters**: Optional filtering by topics, publication_id, dates, etc.

**Similarity Computation**:
```
similarity(query, chunk) = cosine_similarity(query_embedding, chunk_embedding)
```

---

### Phase 5: Relevance Reranking (Classification Model)

#### 5.1 Cross-Encoder Reranker
- **Location**: `core/retrieval.py` → `Reranker` class
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose**: Improve relevance classification by scoring query-document pairs

#### 5.2 Reranking Process

**Step 1: Candidate Expansion**
- Initial retrieval returns `rerank_top_k` candidates (e.g., 10 documents)
- More candidates than final `top_k` to allow reranking to select best matches

**Step 2: Pair Formation**
- For each candidate document, create a query-document pair:
  ```python
  pairs = [
      [query, doc_1_text],
      [query, doc_2_text],
      ...
      [query, doc_10_text]
  ]
  ```

**Step 3: Cross-Encoder Scoring**
- **Model Architecture**: Cross-encoder uses attention mechanism to jointly process query and document
- **Input Format**: `[CLS] query [SEP] document_text [SEP]`
- **Output**: Relevance score (float, typically 0-1 range, higher = more relevant)

**Cross-Encoder Advantages**:
- **Deep Interaction**: Model sees both query and document simultaneously
- **Context-Aware**: Understands semantic relationships beyond embedding similarity
- **Precision**: Better at identifying subtle relevance compared to bi-encoder (embedding) approach

**Step 4: Score-Based Ranking**
- Scores all candidate pairs
- Sorts documents by relevance score (descending)
- Selects top-k documents based on reranking scores

**Reranking Implementation**:
```python
# 1. Retrieve initial candidates
candidates = vectordb.similarity_search(query, k=rerank_top_k)

# 2. Create query-document pairs
pairs = [[query, doc.page_content] for doc in candidates]

# 3. Score pairs with cross-encoder
scores = cross_encoder_model.predict(pairs)

# 4. Sort by score and return top-k
indices = np.argsort(scores)[::-1]  # Descending order
top_results = [candidates[i] for i in indices[:top_k]]
```

#### 5.3 Why Two-Stage Retrieval?

**Stage 1: Vector Search (Bi-Encoder)**
- **Fast**: Efficient approximate nearest neighbor search
- **Scalable**: Can search millions of documents quickly
- **Broad Coverage**: Finds semantically related documents

**Stage 2: Reranking (Cross-Encoder)**
- **Precise**: Deep interaction between query and documents
- **Context-Aware**: Better understanding of query intent
- **Quality**: Improves precision by reordering candidates

**Combined Approach**:
- Fast broad retrieval + precise reranking = Best of both worlds
- Trade-off: Slightly slower but significantly more accurate

---

### Phase 6: Context Assembly and Prompt Construction

#### 6.1 Document Formatting
- **Location**: `llm/prompt_manager.py`
- **Process**:
  1. Formats retrieved documents with publication IDs
  2. Includes excerpts (first ~500 characters) for context
  3. Numbers sources for citation

**Context Block Format**:
```
1. [Publication ID: doc_001]
   Excerpt of document content...

2. [Publication ID: doc_002]
   Excerpt of document content...
```

#### 6.2 Prompt Construction
- **Template**: Qwen instruction format (`<|im_start|>` tokens)
- **Components**:
  1. System prompt (Deutsche Telekom assistant role)
  2. Few-shot examples (safe behavior, citation requirements)
  3. Chat history (if available)
  4. Current query with context documents
  5. Assistant response start token

**Prompt Structure**:
```
<|im_start|>system
You are an enterprise assistant for Deutsche Telekom...
<|im_end|>
<|im_start|>user
Query: [user question]

Context documents:
1. [Publication ID: doc_001]
   [excerpt]
...
<|im_end|>
<|im_start|>assistant
```

---

### Phase 7: Response Generation

#### 7.1 Model Loading
- **Location**: `llm/model_manager.py`
- **Model**: `Qwen/Qwen2.5-3B-Instruct` (production) or `Qwen/Qwen2.5-1.5B-Instruct` (dev)
- **Configuration**:
  - **Dev Mode**: Quantized (int8) for CPU efficiency
  - **Production**: Full precision on GPU, float16 on CUDA

#### 7.2 Generation Parameters
- **max_new_tokens**: 768 (production), 128 (dev)
- **temperature**: 0.6 (production), 0.8 (dev)
- **top_p**: 0.95 (nucleus sampling)
- **Device**: CPU or CUDA (auto-detected)

#### 7.3 Streaming Response
- **Location**: `llm/generation.py` → `generate_response_streaming()`
- **Technique**: TextIteratorStreamer for incremental token generation
- **Benefits**: Real-time feedback to users, perceived faster response

---

## Classification Techniques Summary

### 1. Semantic Classification (Embedding-Based)
- **Method**: Dense vector embeddings
- **Model**: SentenceTransformer (bi-encoder)
- **Purpose**: Initial document retrieval
- **Advantage**: Fast, scalable
- **Limitation**: May miss nuanced relevance

### 2. Relevance Classification (Cross-Encoder)
- **Method**: Query-document pair scoring
- **Model**: Cross-encoder (MS MARCO fine-tuned)
- **Purpose**: Precise relevance ranking
- **Advantage**: High precision, context-aware
- **Limitation**: Slower, requires more computation

### 3. Metadata-Based Classification
- **Method**: Keyword matching and pattern recognition
- **Purpose**: Topic extraction, company identification
- **Use**: Filtering and metadata enrichment
- **Technique**: Regex patterns, keyword buckets

### 4. Hybrid Approach
- **Combination**: Vector search + reranking + metadata filtering
- **Result**: Fast initial retrieval with precise final ranking
- **Trade-off**: Balanced speed and accuracy

---

## Performance Characteristics

### Retrieval Latency
- **Vector Search**: ~10-50ms (depends on collection size)
- **Reranking**: ~50-200ms (depends on rerank_top_k)
- **Total Retrieval**: ~60-250ms

### Accuracy Metrics
- **Recall@K**: Percentage of relevant documents in top-k results
- **Precision@K**: Percentage of top-k results that are relevant
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant document

### Reranking Impact
- **Typical Improvement**: 10-30% improvement in precision@5
- **Trade-off**: ~50-200ms additional latency
- **When to Use**: When accuracy is critical, collection size is manageable

---

## Configuration Parameters

### Chunking
- `CHUNK_SIZE`: 500 characters
- `CHUNK_OVERLAP`: 100 characters

### Retrieval
- `TOP_K`: 5 (final results)
- `RERANK_TOP_K`: 10 (candidates for reranking)
- `RERANKER_MODEL`: "cross-encoder/ms-marco-MiniLM-L-6-v2" (optional)

### Embeddings
- `EMBEDDING_MODEL`: "sentence-transformers/all-MiniLM-L6-v2"
- Dimensions: 384

---

## Conclusion

The RAG system employs a sophisticated two-stage retrieval approach:
1. **Fast Semantic Search**: Vector embeddings for broad recall
2. **Precise Reranking**: Cross-encoder classification for high precision

This hybrid approach balances speed and accuracy, making it suitable for production use while maintaining high-quality relevance classification for document retrieval.
