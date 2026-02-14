"""
DeFrog: RAG for DeFi Documentation
FastAPI backend for querying DeFi protocols
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from psycopg2.extras import RealDictCursor
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
    cache_hit: bool = False


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
            documents_retrieved=result["documents_retrieved"],
            cache_hit=result.get("cache_hit", False)
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _run_ingestion(protocol: Optional[str], force_refresh: bool):
    """Run the ingestion pipeline in a background task"""
    from src.ingestion.crawler import DeFiCrawler
    from src.ingestion.chunking import ChunkingPipeline

    crawler = DeFiCrawler()
    chunker = ChunkingPipeline(chunk_size=800, chunk_overlap=200)

    # Crawl documents (filter by protocol if specified)
    logger.info(f"Starting ingestion for: {protocol or 'all protocols'}")
    all_docs = crawler.crawl_all()

    if protocol:
        all_docs = [d for d in all_docs if d.protocol.lower() == protocol.lower()]

    if not all_docs:
        logger.warning("No documents found to ingest")
        return

    # Chunk and store
    if force_refresh:
        vector_store.clear_all()

    for doc in all_docs:
        chunks = chunker.smart_chunk_text(doc.content, {
            "protocol": doc.protocol,
            "title": doc.title,
            "url": doc.url,
            "doc_type": doc.doc_type
        })
        batch = [{"content": c.content, "metadata": c.metadata} for c in chunks]
        try:
            vector_store.store_documents(batch)
            logger.info(f"Ingested {len(batch)} chunks from {doc.title}")
        except Exception as e:
            logger.error(f"Failed to ingest {doc.title}: {e}")

    logger.info(f"Ingestion complete. Total docs in DB: {vector_store.get_document_count()}")


@app.post("/ingest")
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """Trigger document ingestion pipeline in the background"""
    background_tasks.add_task(_run_ingestion, request.protocol, request.force_refresh)

    return {
        "status": "ingestion_started",
        "protocol": request.protocol or "all",
        "message": "Ingestion pipeline started in background"
    }


def _query_db(sql: str, params: tuple = None) -> list:
    """Execute a read query and return results as list of dicts"""
    conn = vector_store.get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


@app.get("/analytics/costs")
async def get_cost_analytics():
    """Get cost analytics from query_logs table"""
    try:
        rows = _query_db("""
            SELECT
                COUNT(*) as total_queries,
                COALESCE(AVG(cost), 0) as average_cost,
                COALESCE(
                    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::float /
                    NULLIF(COUNT(*), 0), 0
                ) as cache_hit_rate,
                COALESCE(AVG(latency_ms), 0) as avg_latency_ms
            FROM query_logs
        """)

        row = rows[0] if rows else {}
        total_queries = int(row.get('total_queries', 0))

        if total_queries == 0:
            return {
                "total_queries": 0, "average_cost": 0, "cache_hit_rate": 0,
                "avg_latency_ms": 0, "baseline_cost": 0, "optimized_cost": 0
            }

        avg_cost = float(row['average_cost'])
        cache_hit_rate = float(row['cache_hit_rate'])

        # Baseline = avg cost of non-cached queries
        baseline_rows = _query_db("""
            SELECT COALESCE(AVG(cost), 0) as baseline_avg
            FROM query_logs WHERE NOT cache_hit
        """)
        baseline_cost = float(baseline_rows[0]['baseline_avg']) if baseline_rows else avg_cost

        return {
            "total_queries": total_queries,
            "average_cost": round(avg_cost, 6),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "avg_latency_ms": round(float(row['avg_latency_ms']), 1),
            "baseline_cost": round(baseline_cost, 6),
            "optimized_cost": round(avg_cost, 6)
        }
    except Exception as e:
        logger.error(f"Failed to fetch cost analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cost analytics")


@app.get("/analytics/quality")
async def get_quality_metrics():
    """Get quality metrics from evaluation_results table"""
    try:
        rows = _query_db("""
            SELECT
                COUNT(*) as evaluations_run,
                COALESCE(AVG(faithfulness), 0) as faithfulness,
                COALESCE(AVG(answer_relevancy), 0) as answer_relevancy,
                COALESCE(AVG(context_recall), 0) as context_recall
            FROM evaluation_results
        """)

        row = rows[0] if rows else {}
        evaluations_run = int(row.get('evaluations_run', 0))

        if evaluations_run == 0:
            return {
                "faithfulness": 0, "answer_relevancy": 0,
                "context_recall": 0, "evaluations_run": 0
            }

        return {
            "faithfulness": round(float(row['faithfulness']), 4),
            "answer_relevancy": round(float(row['answer_relevancy']), 4),
            "context_recall": round(float(row['context_recall']), 4),
            "evaluations_run": evaluations_run
        }
    except Exception as e:
        logger.error(f"Failed to fetch quality metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quality metrics")


@app.get("/analytics/cache")
async def get_cache_stats():
    """Get live query cache performance statistics"""
    return rag_engine.optimizer.query_cache.get_stats()


@app.post("/evaluate")
async def evaluate_rag_response(request: dict):
    """Evaluate a RAG response and store results in database"""
    from src.evaluation.rag_evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    result = evaluator.evaluate(
        query=request.get("query"),
        answer=request.get("answer"),
        contexts=request.get("contexts", []),
        ground_truth=request.get("ground_truth")
    )

    # Store in database
    conn = None
    cur = None
    try:
        conn = vector_store.get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO evaluation_results
                (experiment_name, query_text, ground_truth, predicted_answer,
                 faithfulness, answer_relevancy, context_recall, cost)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            request.get("experiment_name", "api_evaluation"),
            request.get("query"),
            request.get("ground_truth"),
            request.get("answer"),
            result.faithfulness,
            result.answer_relevancy,
            result.context_recall if result.context_recall >= 0 else None,
            0
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to store evaluation result: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_recall": result.context_recall if result.context_recall >= 0 else None,
        "context_precision": result.context_precision,
        "overall_score": result.overall_score,
        "metadata": result.metadata
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