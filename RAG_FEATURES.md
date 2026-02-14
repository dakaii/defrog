# DeFrog RAG Engineering Features

## RAG Capabilities

A RAG system for DeFi protocol documentation, built with PostgreSQL/pgvector, FastAPI, and multi-LLM support.

## 🎯 Core RAG Competencies

### 1. **Multi-LLM Support**
- ✅ OpenAI GPT models
- ✅ DeepSeek integration
- ✅ Qwen/Dashscope compatibility
- ✅ Configurable base URLs for custom providers

```bash
# Easy provider switching via environment variables
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 2. **Vector Database Engineering**
- **Technology**: PostgreSQL with pgvector extension
- **Indexing**: HNSW for high-performance similarity search
- **Dimensions**: 1536-dim embeddings (configurable)
- **Features**:
  - Metadata filtering
  - Batch ingestion
  - Concurrent query handling

### 3. **Hybrid Search Optimization**
Advanced search combining multiple retrieval methods:

```python
# Dynamic weight optimization based on query characteristics
- Short queries (≤3 words): 50% vector, 50% keyword
- Medium queries (4-7 words): 60% vector, 40% keyword  
- Long queries (>7 words): 75% vector, 25% keyword
```

**Performance Optimizations**:
- Term-overlap reranking
- Query result caching (LRU with TTL)
- Adaptive weight tuning from user feedback

### 4. **RAG Quality Evaluation**
RAGAS-style metrics implementation:

| Metric | Description | Score Range |
|--------|-------------|-------------|
| **Faithfulness** | Measures hallucination | 0.0 - 1.0 |
| **Answer Relevancy** | Query-answer alignment | 0.0 - 1.0 |
| **Context Recall** | Information coverage | 0.0 - 1.0 |
| **Context Precision** | Noise reduction | 0.0 - 1.0 |

### 5. **Production Features**

**Cost Optimization**:
- Query caching system
- Token counting and tracking
- Batch processing support
- Configurable model selection

**Observability**:
- Per-query latency and token usage logged to `query_logs` table
- Cache hit/miss tracking via `/analytics/cache`
- Evaluation results persisted to `evaluation_results` table
- Cost estimation per model

**Deployment**:
- Kubernetes manifests with HPA, liveness/readiness probes
- Pulumi IaC for GCP (Cloud SQL, GKE, Secret Manager)
- Docker Compose for local development

## 📊 Analytics

All metrics are computed from real query and evaluation data stored in PostgreSQL:

```bash
GET /analytics/costs    # Aggregated from query_logs table
GET /analytics/quality  # Aggregated from evaluation_results table
GET /analytics/cache    # Live in-memory cache stats
```

Run evaluations via the `/evaluate` endpoint to populate quality metrics.

## 🧪 Testing

```bash
pytest tests/test_rag_pipeline.py -v
```

## 🔧 API Endpoints

### Query Endpoint
```bash
POST /query
{
  "query": "How does Uniswap concentrated liquidity work?",
  "top_k": 5
}
```

### Evaluation Endpoint
```bash
POST /evaluate
{
  "query": "What is impermanent loss?",
  "answer": "IL occurs when...",
  "contexts": ["context1", "context2"],
  "ground_truth": "optional"
}
```

### Performance Analytics
```bash
GET /analytics/quality
# Returns: faithfulness, relevancy, recall metrics

GET /analytics/costs  
# Returns: token usage, cache stats, cost savings
```

## 🚀 Quick Start

### Local Development
```bash
# With OpenAI
echo "OPENAI_API_KEY=your-key" > .env
docker-compose up

# With DeepSeek
echo "LLM_API_KEY=your-key" > .env
echo "LLM_BASE_URL=https://api.deepseek.com/v1" >> .env
docker-compose up
```

### Run Evaluation
```python
from src.evaluation.rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator()
result = evaluator.evaluate(
    query="Your question",
    answer="RAG response",
    contexts=retrieved_contexts
)
print(f"Faithfulness: {result.faithfulness}")
print(f"Overall Score: {result.overall_score}")
```

## 🏗️ Architecture Highlights

```
┌─────────────────────────────────────┐
│         Query Interface             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Hybrid Search Optimizer        │
│  (Adaptive Weights + Reranking)     │
└──────────────┬──────────────────────┘
               │
     ┌─────────┴─────────┐
     │                   │
┌────▼────┐       ┌──────▼──────┐
│ Vector  │       │   Keyword   │
│ Search  │       │    Search   │
└────┬────┘       └──────┬──────┘
     │                   │
┌────▼───────────────────▼───────────┐
│    PostgreSQL with pgvector        │
│   (HNSW Index + FTS + Metadata)    │
└─────────────────────────────────────┘
```

## 📈 Future Enhancements

- [ ] Connection pooling (currently one connection per request)
- [ ] Real cross-encoder reranking model (currently term-overlap heuristic)
- [ ] Evaluation dataset with ground-truth answers for systematic benchmarking
- [ ] GraphRAG for cross-protocol relationship queries