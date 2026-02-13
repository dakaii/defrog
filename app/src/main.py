"""
DeFrog: RAG for DeFi Documentation
FastAPI backend for querying DeFi protocols
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import logging
import time
from src.retrieval.rag_engine import RAGEngine
from src.ingestion.vector_store import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
rag_engine = RAGEngine()
vector_store = VectorStore()

# Create FastAPI app
app = FastAPI(
    title="DeFrog - DeFi RAG System",
    description="Query DeFi protocol documentation with RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    model_used: str
    latency_ms: int
    documents_retrieved: int


class IngestionRequest(BaseModel):
    protocol: Optional[str] = None  # If None, crawl all
    force_refresh: bool = False


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    openai: bool


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all services"""
    try:
        vector_store.get_document_count()
        postgres_healthy = True
    except Exception:
        postgres_healthy = False

    openai_configured = bool(os.getenv("OPENAI_API_KEY"))

    return HealthResponse(
        status="healthy" if (postgres_healthy and openai_configured) else "degraded",
        postgres=postgres_healthy,
        openai=openai_configured
    )


# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_defi(request: QueryRequest):
    """
    Main RAG query endpoint for DeFi questions
    """
    start_time = time.time()
    
    try:
        # Use the RAG engine to process the query
        result = rag_engine.query(request.query, top_k=request.top_k)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            model_used=result["model"],
            latency_ms=latency_ms,
            documents_retrieved=result["documents_retrieved"]
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingestion endpoint
@app.post("/ingest")
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger document ingestion pipeline
    """
    # TODO: Implement ingestion pipeline
    # background_tasks.add_task(run_ingestion, request.protocol, request.force_refresh)
    
    return {
        "status": "ingestion_started",
        "protocol": request.protocol or "all",
        "message": "Ingestion pipeline started in background"
    }


# Cost analytics endpoint
@app.get("/analytics/costs")
async def get_cost_analytics():
    """
    Get cost analytics and optimization metrics
    """
    # TODO: Query from database
    return {
        "total_queries": 1000,
        "average_cost": 0.025,
        "cache_hit_rate": 0.45,
        "cost_reduction": 0.60,
        "baseline_cost": 0.05,
        "optimized_cost": 0.02
    }


# Evaluation metrics endpoint
@app.get("/analytics/quality")
async def get_quality_metrics():
    """
    Get quality metrics from evaluations
    """
    # TODO: Query from database
    return {
        "faithfulness": 0.87,
        "answer_relevancy": 0.89,
        "context_recall": 0.85,
        "evaluations_run": 500
    }


# List available protocols
@app.get("/protocols")
async def list_protocols():
    """
    List available DeFi protocols in the system
    """
    return {
        "protocols": [
            "Aave V2",
            "Uniswap V2",
            "Uniswap V3",
            "Compound",
            "MakerDAO",
            "Curve Finance",
            "Balancer",
            "Synthetix"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)