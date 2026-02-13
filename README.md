# 🐸 DeFrog - DeFi Protocol RAG System

> **Clearing the fog around DeFi data** - A RAG system for querying DeFi protocol documentation and whitepapers.

## 🎯 Overview

DeFrog is a Retrieval Augmented Generation (RAG) system that allows you to query DeFi protocol documentation using natural language. It ingests whitepapers from major DeFi protocols and uses vector search + LLMs to provide accurate, sourced answers.

## ✨ Features

- **Vector Search**: Uses pgvector for semantic search through DeFi documentation
- **Multiple Protocols**: Includes Uniswap, Aave, Compound, MakerDAO, Curve, and more
- **Source Attribution**: Every answer includes sources from the original documentation
- **Simple Interface**: Clean Streamlit UI for easy querying
- **Docker Ready**: Full Docker Compose setup for easy deployment

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
git clone <repo-url>
cd defrog
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Start the services**
```bash
docker-compose up -d
```

4. **Ingest DeFi documents**
```bash
# First run: ingest without clearing
docker-compose exec app python scripts/ingest_defi_docs.py

# To re-ingest from scratch (clears existing documents):
docker-compose exec app python scripts/ingest_defi_docs.py --clear
```

5. **Access the UI**
```bash
open http://localhost:8501
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         PostgreSQL + pgvector       │
│         (Vector Database)           │
└──────────────▲──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│          FastAPI Backend            │
│    (RAG Engine + API Endpoints)     │
└──────────────▲──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Streamlit Frontend           │
│        (Query Interface)            │
└─────────────────────────────────────┘
```

## 📚 Included DeFi Protocols

- **Uniswap V2 & V3** - Automated Market Maker
- **Aave** - Lending Protocol
- **Compound** - Lending Market
- **MakerDAO** - Stablecoin Protocol
- **Curve Finance** - Stableswap
- **Balancer** - Weighted Pools
- **Synthetix** - Synthetic Assets

## 💻 API Endpoints

### Query Endpoint
```bash
POST /query
{
  "query": "How does Uniswap V3 concentrated liquidity work?",
  "top_k": 5
}
```

### Health Check
```bash
GET /health
```

### List Protocols
```bash
GET /protocols
```

## 🛠️ Tech Stack

- **Database**: PostgreSQL with pgvector extension
- **Backend**: FastAPI + Python 3.11
- **Frontend**: Streamlit
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4o-mini
- **Infrastructure**: Docker Compose

## 📁 Project Structure

```
defrog/
├── app/                    # FastAPI backend
│   ├── src/
│   │   ├── ingestion/     # Document crawler & chunking
│   │   ├── retrieval/     # RAG engine
│   │   └── main.py        # API endpoints
│   └── scripts/           # Ingestion scripts
├── dashboard/             # Streamlit UI
├── data/                  # Data storage
├── scripts/               # Database init
└── docker-compose.yml     # Container setup
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# OpenAI
OPENAI_API_KEY=your_key_here

# Model Selection
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Database
POSTGRES_DB=defrog
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

## 📝 Example Queries

- "How does Uniswap V3 concentrated liquidity work?"
- "What is the difference between Uniswap V2 and V3?"
- "How does Aave's liquidation mechanism work?"
- "Explain Compound's interest rate model"
- "What are the risks of providing liquidity in Curve?"
- "How does MakerDAO maintain DAI's peg?"

## 🚢 Deployment

### Local Development
```bash
docker-compose up
```

### Production (Future)
Ready for deployment to:
- GCP with Cloud SQL + Cloud Run
- AWS with RDS + ECS
- Any Kubernetes cluster

## 📜 License

MIT

---

Built for exploring DeFi protocols through natural language queries.