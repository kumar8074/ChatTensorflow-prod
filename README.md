# ChatTensorFlow 🔗

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-FF6B6B?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.10.4-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.19.0-005EB8?style=for-the-badge&logo=opensearch&logoColor=white)](https://opensearch.org)
[![LangSmith](https://img.shields.io/badge/LangSmith-Latest-5F66F6?style=for-the-badge&logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Latest-1D63ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-Latest-FA6E32?style=for-the-badge&logo=uv&logoColor=white)](https://github.com/astral-sh/uv)

> **A production-grade RAG system for TensorFlow documentation that thinks, plans, and delivers—no hand-holding required.**

Most RAG implementations answer questions. ChatTensorFlow understands them. We built an intelligent assistant that routes queries, plans research strategies, and streams answers with proper citations—backed by a self-sustaining pipeline that keeps your knowledge fresh while you sleep.

---

## What Makes This Different

### Infrastructure-First, Always
No "we'll fix it in production" promises. ChatTensorFlow ships with Docker Compose orchestration, health checks, resource management, and battle-tested reliability. The system scales, recovers from failures, and monitors itself. This is production infrastructure, not a weekend project.

### A Pipeline That Never Sleeps
Airflow orchestrates monthly ingestion runs that crawl TensorFlow's entire documentation tree using sitemap-based discovery. The pipeline chunks intelligently, preserves code blocks with context, generates embeddings in batches, and indexes everything into OpenSearch—automatically. Set the schedule and walk away.

### Hybrid Search With Intelligence
BM25 lexical matching meets vector similarity through Reciprocal Rank Fusion. Query-type detection dynamically adjusts field boosting—code snippets rise to the top for implementation questions, API references dominate for parameter queries, tutorials surface for how-to questions. Search that feels like it reads your mind.

### An Agent That Actually Plans
LangGraph powers an assistant that doesn't just retrieve and regurgitate. It classifies query intent, generates multi-step research plans, executes parallel document retrieval across the TensorFlow documentation hierarchy, and synthesizes answers with full conversation memory. This is agentic RAG done right.

### Real-Time Streaming, Not Spinners
Server-Sent Events stream every node execution in the graph—not just the final answer. Users see the system analyze queries, create research plans, retrieve documents, and generate responses in real-time. Progress indicators replace loading spinners. Transparency builds trust.

### Citations That Matter
Every claim links directly to TensorFlow's documentation with clean, context-aware citations. URLs get shortened intelligently to show meaningful paths (e.g., `[tf.keras.layers.Dense]` instead of the full URL). Breadcrumbs reveal exactly where information lives in the docs hierarchy.

### Observability From Day One
LangSmith integration traces every LLM call, embedding generation, and retrieval operation. Track token usage, latency, conversation flows, and debug paths through the graph. When something breaks at 3 AM, you'll know exactly what happened and why.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (MONTHLY)                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │ Scraper  │ → │ Chunker  │ → │ Embedder │ → │ Indexer  │      │
│  │(Sitemap) │   │(Context) │   │ (Batch)  │   │(Stream)  │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│       ↓              ↓              ↓              ↓            │
│   Crawl4AI      Smart Split    768-dim Vec    OpenSearch        │
│   + Sitemap     Code Blocks    Text-004       k-NN Index        │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY PROCESSING                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │  Router  │ → │ Planner  │ → │Researcher│ → │Generator │      │
│  │(Classify)│   │(Strategy)│   │(Parallel)│   │ (Stream) │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│       ↓              ↓              ↓              ↓            │
│   3-way Split   1-3 Steps     RRF Fusion     SSE Stream         │  
│   TF/More/Gen   Execution     Top-K Docs    +Citations          │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │LangSmith │   │  Logs    │   │ Metrics  │   │Checksums │      │
│  │ (Traces) │   │(Rotation)│   │ (Health) │   │(Memory)  │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Tech Stack

**Orchestration**
- Docker Compose with health checks and auto-restart policies
- Apache Airflow for scheduled ingestion (runs monthly on the 27th at 11 PM IST)

**Storage & Search**
- OpenSearch 2.19 with k-NN plugin for vector similarity
- PostgreSQL 16 for Airflow metadata and LangGraph checkpointing

**Document Processing**
- Crawl4AI with Playwright for JavaScript-rendered documentation
- Sitemap-based discovery with intelligent filtering
- Custom chunker preserving code blocks, headings, and contextual relationships
- Google Gemini embeddings (text-embedding-004, 768 dimensions)

**Intelligence Layer**
- LangGraph for stateful agent workflows with memory and checkpointing
- Gemini 2.5 Flash for rapid inference and structured outputs
- Custom hybrid search with dynamic query-type detection and field boosting

**API & Frontend**
- FastAPI with SSE streaming and comprehensive request validation
- Custom JavaScript chat interface with localStorage thread persistence
- Prism.js for syntax highlighting with one-click copy functionality

**Monitoring**
- LangSmith for complete LLM observability and conversation tracing
- Structured logging with automatic rotation and timestamps
- OpenSearch Dashboards for search analytics and index health

---

## The Tech Stack (Continued)

**Key Design Decisions:**
- **Sitemap-first scraping**: Follows TensorFlow's official documentation structure, respects robots.txt, handles pagination automatically
- **Streaming indexing**: Uses `ijson` to process large JSON files without loading everything into RAM
- **Query-aware retrieval**: Detects whether users want code examples, API docs, or conceptual guides—then adjusts search accordingly
- **Memory-efficient architecture**: Processes embeddings in batches, streams data to OpenSearch, uses checkpoint-based conversation persistence

---

## Project Structure

```
chattensorflow/
│
├── airflow/                          # Airflow orchestration
│   ├── dags/
│   │   ├── tf_ingestion.py           # Main DAG definition
│   │   └── ingestion_tasks/          # Task implementations
│   │       ├── scrapper_task.py      # Sitemap-based crawler
│   │       ├── chunking_task.py      # Smart document chunking
│   │       ├── embedding_task.py     # Batch embedding generation
│   │       └── indexing_task.py      # Streaming OpenSearch indexer
│   └── Dockerfile                    # Airflow custom image with Playwright
│
├── src/                              # Core application
│   ├── config.py                     # Pydantic-based configuration
│   ├── logger.py                     # Structured logging setup
│   ├── dependencies.py               # FastAPI dependency injection
│   ├── main.py                       # FastAPI application entry point
│   │
│   ├── services/                     # Business logic layer
│   │   ├── scrapper/
│   │   │   └── tensorflow_scrapper.py   # Async sitemap crawler with Crawl4AI
│   │   ├── chunking/
│   │   │   └── content_chunker.py    # Context-aware document chunking
│   │   ├── embedding/
│   │   │   └── embedding_service.py  # Batch embedding with progress tracking
│   │   ├── opensearch/
│   │   │   ├── factory.py            # Connection pooling and health checks
│   │   │   └── hybrid_search_service.py  # BM25 + Vector with RRF
│   │   ├── indexing/
│   │   │   ├── index_config.py       # TensorFlow-optimized mappings
│   │   │   └── opensearch_indexer.py # Streaming bulk indexer
│   │   └── rag/
│   │       ├── states.py             # LangGraph state schemas
│   │       ├── prompts.py            # System and routing prompts
│   │       ├── researcher_subgraph.py # Parallel retrieval subgraph
│   │       ├── tf_graph.py           # Main agent graph with memory
│   │       └── rag_service.py        # High-level RAG orchestration
│   │
│   ├── routers/
│   │   └── rag.py                    # FastAPI endpoints with SSE
│   │
│   ├── schemas/
│   │   └── api/
│   │       └── rag.py                # Pydantic request/response models
│   │
│   └── frontend/                     # Chat interface
│       ├── index.html                # Single-page application
│       ├── style.css                 # Modern dark theme
│       └── chat.js                   # Vanilla JS with localStorage
│
├── temp/                             # Pipeline temporary storage
│   ├── docs_rag.json                 # Raw scraped documentation
│   └── chunked_data/                 # Processed documents
│       ├── chunks_for_rag.jsonl      # JSONL chunks for streaming
│       ├── all_chunks.json           # Complete chunk data
│       ├── chunks_with_embeddings.json  # Embedded chunks
│       └── chunking_statistics.json  # Ingestion metrics
│
├── logs/                             # Application logs
│   └── *.log                         # Timestamped log files
│
├── compose.yml                       # Multi-service orchestration
├── Dockerfile                        # Multi-stage FastAPI build
├── pyproject.toml                    # Python dependencies (uv)
├── uv.lock                           # Locked dependency versions
├── .env.example                      # Environment template
└── README.md                        
```

**Key Directories Explained:**

- **`airflow/dags/ingestion_tasks/`**: Each task is a standalone module. Tasks can be tested independently, composed into different DAGs, or run manually via `airflow tasks test`.

- **`src/services/rag/`**: The brain of the system. `tf_graph.py` defines the entire agent workflow—query routing, research planning, parallel retrieval, response generation, and conversation summarization. State management uses LangGraph's checkpointing for persistence.

- **`src/services/opensearch/`**: Hybrid search implementation with query-type detection. The system automatically adjusts field boosting based on whether users ask about code, APIs, or concepts.

- **`src/frontend/`**: Zero-framework vanilla JavaScript. Thread management via `localStorage`, real-time SSE streaming, automatic code highlighting, and citation formatting. No build step required.

- **`temp/`**: Airflow writes intermediate data here. Raw scraped content, processed chunks, and embeddings live temporarily before indexing. Useful for debugging pipeline issues.

---

## What It Actually Does

### Intelligent Document Ingestion
The scraper doesn't just crawl—it reads TensorFlow's sitemap, filters out non-Python content (JavaScript, Swift, C++ docs), validates URLs against inclusion/exclusion patterns, and uses Crawl4AI with Playwright to handle JavaScript-rendered pages. The chunker then extracts code blocks separately, maintains heading hierarchies, creates overlapping windows for context continuity, and enriches every chunk with breadcrumbs and page type metadata.

### Query-Aware Retrieval
Ask "How to build a CNN model?" and the system detects it's a code-oriented query. It boosts `code_blocks.code` and `full_text` fields, prioritizes tutorial and example pages, and adjusts RRF weights to favor code snippets. Ask "What are the parameters of Dense layer?" and it pivots to API reference pages with heading and title emphasis. The search adapts to your intent.

### Research Planning That Works
The assistant analyzes complex queries and breaks them into 1-3 concrete research steps. Each step generates diverse search queries (no repetitive variations), retrieves documents in parallel using LangGraph's `Send()` nodes, and accumulates knowledge before formulating the final answer. Multi-hop reasoning without the hand-waving.

### Conversation Memory That Persists
Every thread uses LangGraph's checkpointing for persistent memory. When context exceeds 1000 tokens, the system automatically summarizes older messages while keeping the last 3 for immediate context. Users can close their browser, come back days later, and resume exactly where they left off—with full conversation history intact.

### Source Attribution Done Right
Responses include citations formatted as `[tf.keras.layers.Dense]` that link directly to the source documentation. URLs are shortened intelligently using TensorFlow's path structure and anchor text. Breadcrumb trails show the exact documentation hierarchy. Users can verify every claim with one click.

---

## Running It Locally

**Prerequisites**
- Docker Desktop with 8GB+ RAM allocated
- Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))
- LangSmith API key for observability ([free tier](https://smith.langchain.com))

**Quick Start**
```bash
# Clone and navigate
git clone https://github.com/kumar8074/chattensorflow.git
cd chattensorflow

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start everything
docker compose up -d

# Watch the magic happen
docker compose logs -f fastapi-app
```

**Access Points**
- Chat Interface: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Airflow UI: `http://localhost:8080` (admin/admin)
- OpenSearch: `http://localhost:9200`
- OpenSearch Dashboards: `http://localhost:5601`

**First-Time Setup**
The system starts with an empty index. Trigger the ingestion pipeline:

1. Open Airflow at `http://localhost:8080`
2. Enable the `tensorflow_documentation_ingestion` DAG
3. Click "Trigger DAG" (takes ~45-60 minutes for full TensorFlow docs)
4. Monitor progress in the Graph View

Alternatively, run the pipeline steps manually:
```bash
# Enter the Airflow scheduler container
docker exec -it chatTF-airflow-scheduler bash

# Run the full pipeline
airflow dags test tensorflow_documentation_ingestion
```

---

## Configuration That Matters

**Chunking Strategy** (`src/config.py`)
```python
CHUNK_SIZE = 1000        # Words per chunk
CHUNK_OVERLAP = 200      # Overlap for continuity
```
Smaller chunks = more precise retrieval, higher API costs  
Larger chunks = better context preservation, fewer fragments

**Hybrid Search Weights** (`src/services/opensearch/hybrid_search_service.py`)
```python
bm25_weight = 0.4        # Lexical matching importance
vector_weight = 0.6      # Semantic similarity importance
```
Increase BM25 for exact function name matching  
Increase vector weight for conceptual question understanding

**Retrieval Top-K**
```python
top_k = 5               # Documents per query
```
More documents = comprehensive answers, slower responses  
Fewer documents = faster replies, risk missing relevant context

**Airflow Schedule** (`airflow/dags/tf_ingestion.py`)
```python
schedule_interval = '0 23 27 * *'  # 11 PM IST on 27th monthly
```
Adjust frequency based on TensorFlow documentation update cadence

---

## The API Contract

**POST /api/rag/ask**  
Execute a query and return the complete result.
```json
{
  "user_query": "How to implement a custom training loop?",
  "user_id": "user_123",
  "thread_id": "thread_456"
}
```
Response includes `response`, `metadata` (sources, research steps, router info), and `status`.

**POST /api/rag/ask/stream**  
Execute with real-time streaming via Server-Sent Events.
```javascript
fetch('/api/rag/ask/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_query, user_id, thread_id })
}).then(response => {
  const reader = response.body.getReader();
  // Process SSE chunks: node events, response chunks, completion
});
```
Streams `analyze_and_route_query`, `create_research_plan`, `conduct_research`, `respond`, and `summarize_conversation` node updates.

**POST /api/rag/history**  
Retrieve conversation history with optional message limit.
```json
{
  "user_id": "user_123",
  "thread_id": "thread_456",
  "limit": 20
}
```

**DELETE /api/rag/history**  
Clear all messages and state for a conversation thread.

---

## Production Considerations

**Resource Requirements**
- Minimum: 8GB RAM, 4 CPUs
- Recommended: 16GB RAM, 8 CPUs
- Storage: ~3GB for TensorFlow docs index + embeddings

**Scaling Strategies**
- Increase Airflow `PARALLELISM` for faster multi-document ingestion
- Add FastAPI workers with `--workers N` in Dockerfile CMD
- Deploy OpenSearch cluster mode with replica shards
- Use Redis for distributed LangGraph checkpointing

**Security Hardening**
- Enable OpenSearch security plugin with TLS in production
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Implement rate limiting on FastAPI endpoints
- Add authentication middleware for the chat interface

**Monitoring in Production**
- Set up LangSmith alerts for high latency or errors
- Configure OpenSearch slow query logs and index health alerts
- Monitor Airflow DAG success rates and set up SLA notifications
- Track Gemini API quota usage and set billing alerts

---

## The Details That Matter

**Why Sitemap-Based Scraping?**  
Following the official sitemap respects TensorFlow's documentation structure, avoids crawling deprecated pages, handles pagination automatically, and ensures comprehensive coverage without manually maintaining URL lists. It's self-updating—when TensorFlow adds new docs, the scraper finds them.

**Why Streaming Indexing?**  
Loading a 500MB JSON file with embeddings into RAM kills containers. Streaming with `ijson` reads one chunk at a time, processes in batches, and sends to OpenSearch incrementally. Memory usage stays constant regardless of dataset size.

**Why Hybrid Search?**  
Pure vector search fails on exact function names like `tf.keras.layers.Dense`. Pure BM25 misses semantic equivalence like "neural network architecture" vs "model structure". Hybrid with RRF gives you both—lexical precision and semantic understanding.

**Why LangGraph?**  
Linear RAG chains can't handle multi-turn conversations with context, complex queries requiring multiple retrieval steps, or conditional logic based on query type. LangGraph's state machines enable true agentic behavior—planning, branching, looping, and memory.

**Why Streaming?**  
Users abandon requests after 10 seconds of silence. Streaming provides immediate feedback (the system is working), enables progressive information consumption (read while generating), and improves perceived performance. Psychology matters in UX.

**Why LangSmith?**  
Production LLM systems fail in mysterious ways. LangSmith captures every prompt, completion, token count, latency, and graph execution path. When users report incorrect answers, you see exactly which documents were retrieved, how they were ranked, and what prompts were sent to the LLM.

---

## What's Next

This system ships production-ready, but there's always room to push further:

- **Multi-model support**: Let users switch between Gemini, Claude, and GPT-4 for responses
- **Document upload**: Extend beyond TensorFlow to user-uploaded code files
- **Feedback loops**: Implement thumbs up/down to fine-tune retrieval ranking
- **Collaborative features**: Share threads publicly, annotate responses, create knowledge bases
- **Cost optimization**: Implement semantic caching to reduce redundant LLM calls

A Simple, No infrastructure version of this project can be accessed [Here](https://github.com/kumar8074/chatTensorFlow)

---

## Built By

**LALAN KUMAR**  
[GitHub](https://github.com/kumar8074) | [LinkedIn](https://www.linkedin.com/in/lalan-kumar-983267229/)

---

## License

MIT License

---

**Remember:** Most teams ship RAG systems that hallucinate, lose context, and break under load. We ship RAG with query-aware retrieval, persistent memory, streaming responses, full observability, and production-grade infrastructure. That's the difference between a demo and a system you can bet your product on.

Now go ship something that actually works.
