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
  - `chunk_size`: 800 characters (default)
  - `chunk_overlap`: 150 characters (default)
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
- **Model**: `intfloat/multilingual-e5-large`
- **Dimensions**: 1024-dimensional vectors
- **Rationale**: 
  - Multilingual support for better cross-language understanding
  - High-quality embeddings for improved retrieval accuracy
  - Compatible with LangChain's embedding interface

#### 2.2 Embedding Generation
- **Process**:
  - Each chunk is passed through the SentenceTransformer model
  - Generates dense vector representations capturing semantic meaning
  - Vectors are normalized and stored for similarity computation

**Embedding Properties**:
- **Semantic Similarity**: Captures meaning, not just keyword matching
- **Dimensionality**: 1024 dimensions per chunk
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
  - Embedding vectors (1024-dim)
  - Document metadata (publication_id, topics, dates, etc.)
  - Chunk identifiers and indices
  - Source text (for retrieval)

---

### Phase 4: Query Processing and Initial Retrieval

#### 4.1 Query Embedding
- **Process**:
  1. User query is received via API or UI
  2. Same embedding model encodes query into 1024-dimensional vector
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
  2. Includes excerpts (first ~800 characters) for context
  3. Numbers sources for citation

**Context Block Format**:
```
1. [Publication ID: doc_001]
   Excerpt of document content...

2. [Publication ID: doc_002]
   Excerpt of document content...
```

#### 6.2 Prompt Construction
- **Template**: Hugging Face chat template (LLaMA/Gemma/Mistral compatible)
- **Components**:
  1. System prompt (Deutsche Telekom assistant role)
  2. Few-shot examples (safe behavior, citation requirements)
  3. Chat history (if available)
  4. Current query with context documents
  5. Assistant response start token

**Prompt Structure**:
The prompt is built using Hugging Face's `apply_chat_template()` method, which generates a model-specific format. The message structure follows this pattern:
- System message with instructions
- Few-shot behavioral examples (user/assistant pairs)
- Chat history (if available)
- Current user query with context documents
- Assistant response generation prompt

The tokenizer automatically formats messages according to the model's chat template (e.g., Gemma 2 uses its own chat template format compatible with Hugging Face's chat template system).

---

### Phase 7: Response Generation

#### 7.1 Model Loading
- **Location**: `llm/model_manager.py`
- **Model**: `google/gemma-3-4b-it` (production) or `google/gemma-3-1b-it` (dev)
- **Configuration**:
  - **Dev Mode**: Quantized (int8) for CPU efficiency
  - **Production**: Full precision on GPU, float16 on CUDA

#### 7.2 Generation Parameters
- **max_new_tokens**: 768 (production), 128 (dev)
- **temperature**: 0.6 (production), 0.8 (dev)
- **top_p**: 0.9 (nucleus sampling)
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
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 150 characters

### Retrieval
- `TOP_K`: 5 (final results)
- `RERANK_TOP_K`: 10 (candidates for reranking)
- `RERANKER_MODEL`: "cross-encoder/ms-marco-MiniLM-L-6-v2" (optional)

### Embeddings
- `EMBEDDING_MODEL`: "intfloat/multilingual-e5-large"
- Dimensions: 1024

---

## Optimization and Model Iteration

Throughout the development of this RAG system, extensive experimentation has been conducted to improve response quality, naturalness, and user interaction experience. Multiple configurations have been tested across different dimensions:

### Model Iterations

The system has been evaluated with several model families to find the optimal balance between response quality, coherence, and computational efficiency. The following model pairs have been tested:

1. **Qwen Family (Initial)**
   - Production: `Qwen/Qwen2.5-3B-Instruct`
   - Development: `Qwen/Qwen2.5-1.5B-Instruct`

2. **Qwen Family (Enhanced)**
   - Production: `Qwen/Qwen3-4B-Instruct-2507`
   - Development: `Qwen/Qwen2.5-1.5B-Instruct`

3. **Meta LLaMA Family**
   - Production: `meta-llama/Meta-Llama-3-8B-Instruct`
   - Development: `meta-llama/Meta-Llama-3-1B-Instruct`

4. **Google Gemma-2 Family**
   - Production: `google/gemma-2-9b-it`
   - Development: `google/gemma-2-2b-it`

5. **Google Gemma-3 Family (Current)**
   - Production: `google/gemma-3-4b-it`
   - Development: `google/gemma-3-1b-it`

### Chunking Parameter Optimization

Chunking parameters have been adjusted to improve context retrieval and response quality:

- **CHUNK_SIZE**: Experimented with values from 500 to 800 characters
  - Initial: 500 characters
  - Current: 800 characters (provides more context per chunk)
  
- **CHUNK_OVERLAP**: Adjusted to maintain context continuity
  - Initial: 100 characters
  - Current: 150 characters (better overlap ratio for larger chunks)

The larger chunk size and increased overlap were found to provide more comprehensive context to the language model, enabling more coherent and informative responses.

### Generation Parameter Tuning

Temperature and top_p parameters have been fine-tuned to balance creativity with consistency:

- **TEMPERATURE**: Maintained at 0.6 (production) - after several attempt at lower values - for balanced creativity and consistency
- **TOP_P**: Adjusted from 0.95 to 0.9 to reduce randomness while maintaining natural language flow
- **MAX_CONTEXT_TOKENS**: Increased from 2000 to 6000 to allow more comprehensive context from retrieved documents, enabling richer and more detailed responses

These settings were optimized to produce responses that are both informative and natural-sounding, avoiding overly formal or robotic language while maintaining factual accuracy.

### Prompt Engineering Evolution

The prompt construction approach has evolved from model-specific formatting (Qwen-style `<|im_start|>` tokens) to a universal Hugging Face chat template system. This change:

- Enables compatibility across multiple model families (LLaMA, Gemma, Mistral)
- Leverages native tokenizer chat templates for optimal formatting
- Provides consistent behavior regardless of the underlying model architecture

### Response Quality Improvements

The iterative optimization process has focused on:

- **Naturalness**: Making responses sound more conversational and less structured
- **Coherence**: Ensuring responses flow naturally and synthesize information effectively
- **Completeness**: Providing comprehensive answers that fully address user queries
- **Friendliness**: Creating a warm, engaging interaction experience while maintaining professionalism

Current configuration (Gemma-3-4B with chunk size 800, overlap 150, temperature 0.6, top_p 0.9) represents the culmination of these optimization efforts, providing the best balance of response quality, naturalness, and user experience. The migration from Gemma-2 to Gemma-3 models provides improved performance and efficiency while maintaining the same high-quality response characteristics but also utilising less resources than the previous generation.

---

## Conclusion

The RAG system employs a sophisticated two-stage retrieval approach:
1. **Fast Semantic Search**: Vector embeddings for broad recall
2. **Precise Reranking**: Cross-encoder classification for high precision

This hybrid approach balances speed and accuracy, making it suitable for production use while maintaining high-quality relevance classification for document retrieval.


# System Interface

## Overview

The RAG system provides multiple interaction layers tailored for different operational and development needs. Each interface serves a distinct purpose within the overall architecture, balancing usability, flexibility, and scalability.  

Currently, three main interaction methods are available:

1. **Streamlit Application** — Internal experimentation and parameter tuning  
2. **FastAPI REST Gateway** — Primary programmatic interface  
3. **React Chatbot Application** — End-user facing web interface  

---

## 1. Streamlit Application (`ui`)

### Purpose
The Streamlit interface serves as an **internal research and debugging tool**, enabling rapid experimentation with RAG parameters and configurations. It provides an accessible environment for data scientists and engineers to test retrieval and generation behavior interactively before promoting configurations to production.

### Characteristics
- Designed for **scientific experimentation and parameter optimization**
- Supports **streaming output**, simulating real-time inference behavior
- Minimal setup; optimized for **local development and validation**
- Directly integrates with the core RAG logic (retrieval, reranking, generation)

### Limitations
- Not intended for production use or large-scale deployment  
- Limited scalability and customization capabilities compared to the API layer

### Use Case
Ideal for:
- Rapid prototyping and model comparison
- Debugging retrieval and generation logic
- Demonstrating internal proof-of-concepts

---

## 2. FastAPI REST Gateway (`api`)

### Purpose
The FastAPI gateway represents the **primary access layer** for exposing RAG capabilities via a **RESTful API**. It acts as the operational backbone of the system, enabling both internal and external applications to interact with the RAG pipeline in a standardized and scalable manner.

### Core Endpoints
- **`/query`** — Main inference endpoint that processes user queries and returns RAG-generated responses.
- **Health & Monitoring** — Auxiliary endpoints for system health checks, metrics, and status reporting.

### Characteristics
- Serves as the **main production entry point**
- Provides structured **JSON-based responses**
- Easily extensible with **authentication**, **rate limiting**, and **multi-feature support**
- Fully compatible with the React frontend and other potential clients

### Streaming
Currently, responses are returned as **non-streaming** payloads.  
Future iterations may include **token-level streaming** for real-time conversational experiences.

### Advantages
- Flexible configuration and deep integration options  
- Production-ready scalability and monitoring capabilities  
- Compatible with container orchestration and CI/CD pipelines  

### Use Case
Preferred for:
- Integrating RAG capabilities into other services  
- Building enterprise or multi-client applications  
- Production deployment and monitoring scenarios  

---

## 3. React Chatbot Application (`react`)

### Purpose
The React chatbot provides a **user-facing conversational interface** for interacting with the RAG API. It is designed to simulate a realistic end-user experience and serves as a **reference implementation** for client-side integration.

### Characteristics
- Lightweight **Single Page Application (SPA)** built with React  
- Connects directly to the FastAPI `/query` endpoint  
- Provides a conversational chatbot-style interface  
- Includes minimal configuration for **easy deployment and maintenance**

### Deployment
- Deployed automatically via **GitHub Actions** once Python CI tests pass  
- Hosted using **GitHub Pages**  
- Environment variables securely injected through **GitHub Secrets**

### Purpose and Scope
While not designed as a full-scale production frontend, it offers:
- A functional demonstration of RAG’s user interaction flow  
- A baseline implementation for future UX/UI extensions  
- Quick local or remote deployment with minimal configuration overhead  

### Limitations
- Not optimized for enterprise-grade scaling  
- Simplified state management and architecture for demonstration purposes  

### Use Case
Ideal for:
- User acceptance testing  
- Internal demos and stakeholder presentations  
- Rapid feedback cycles on RAG system behavior  

---

# Deployment and Execution

## Overview
Multiple deployment and execution options are available to support diverse development and production workflows. These methods ensure flexibility across environments—from local prototyping to GPU-accelerated cloud deployment.

---

## 1. Local Execution (Python / Poetry)
The system can be executed directly using Python or Poetry for quick iteration and debugging.  
This mode is recommended for:
- Local experimentation
- Development of new ingestion or retrieval features
- Unit and integration testing

---

## 2. Makefile Commands
A dedicated `Makefile` provides simplified command shortcuts for routine operations such as:
- Running the API or UI locally
- Setting up and running the React application locally
- Debugging and preparing the python applications
- Rebuilding vector indexes
- Running ingestion and cleaning datasets
- Debbuging and running the docker images
This ensures **consistent developer experience** and **standardized workflows**.

---

## 3. Docker-Based Development

### Local Development (`docker-compose.dev.yml`)
- Designed for **hot-reload development**  
- Runs inference on **CPU** (optimized for compatibility, not performance)  
- Requires manual ingestion (via shell access to the container)  
- Intended for iterative local development and debugging
- Recommended specs: ≥6 CPU cores, ≥7GB RAM (≥5GB if using dtype float16)

### Staging Environment (`docker-compose.yml`)
- Targets **GPU-backed systems** (e.g., NVIDIA T4 or higher)  
- Includes an additional **worker container** dedicated to rebuilding Chroma indexes  
- Allows dataset updates without downtime in API/UI services  
- Recommended GPU specs: ≥14GB VRAM, CUDA 12.x+

### Notes
- Staging environment can also run locally on compatible hardware  
- Index rebuilds are triggered manually (future improvement: scheduled cron-based jobs)

---

## 4. Production Image Deployment

### Description
A **standalone Docker image** is used for production-grade deployment, focusing on **portability and simplicity**.  
It is pushed to a **public DockerHub repository** and deployed on **Vast.ai GPU instances** for accelerated inference.

### Execution Flow
- On container startup, ingestion and indexing are triggered automatically via `entrypoint.sh`
- Self-signed SSL certificates are used for FastAPI (as a proof of concept)
- Future enhancements include:
  - Integration with CI/CD pipelines
  - Automated SSL and secure secrets management
  - Scalable cloud-native deployment (Kubernetes-ready)

### Performance Note
While CPU-based inference is functional, **GPU-backed instances** are strongly recommended for production use due to the significant performance difference.

---

## 5. React Application Deployment
The React chatbot (`react/`) is **not containerized** within the Docker setups.  
It is built and deployed separately through the GitHub CI/CD workflow, as described above.

---

## Summary

| Deployment Mode | Purpose | Inference Type | Notes |
|------------------|----------|----------------|-------|
| **Local (Python/Poetry)** | Quick debugging & dev | CPU | Fast iteration |
| **Makefile** | Consistent dev ops | CPU/GPU | Simplifies commands |
| **Docker (Dev)** | Local hot-reload | CPU | Manual ingestion |
| **Docker (Staging)** | Pre-production testing | GPU | Worker-based indexing |
| **Production Image** | Live deployment | GPU | Vast.ai or similar |
| **React SPA** | User interface | — | Deployed via GitHub Pages |

---

## Conclusion on Interface, Deployment & Execution/Development

The RAG system exposes a robust and flexible interface architecture:
- A **Streamlit UI** for experimentation and parameter tuning  
- A **FastAPI gateway** for scalable production-level integration  
- A **React chatbot** for intuitive human interaction  

Together with multi-environment deployment options—from local CPU testing to GPU-backed production—the system offers both agility during development and reliability during operation.
