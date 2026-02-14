"""
RAG engine for DeFi queries with optimized hybrid search and query logging
"""
import os
import time
import logging
from typing import List, Dict
from openai import OpenAI
from src.ingestion.vector_store import VectorStore
from src.optimization.search_optimizer import HybridSearchOptimizer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pricing per 1K tokens (used for cost estimation)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "deepseek-chat": {"input": 0.00014, "output": 0.00028},
}


class RAGEngine:
    """RAG implementation for DeFi queries with optimized search and logging"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.optimizer = HybridSearchOptimizer(self.vector_store)

        # Support different LLM providers
        api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('LLM_BASE_URL')  # For DeepSeek, Qwen, etc.

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = os.getenv('LLM_MODEL', 'gpt-4o-mini')

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Process a query using RAG:
        1. Retrieve relevant documents via optimized hybrid search
        2. Build context
        3. Generate answer using LLM
        4. Log query metrics
        """
        start_time = time.time()

        # Step 1: Retrieve via optimizer (with caching)
        logger.info(f"Retrieving documents for: {question}")
        search_results = self.optimizer.adaptive_search(question, top_k=top_k)
        cache_hit = self.optimizer.was_cache_hit()
        documents = self.optimizer.search_results_to_dicts(search_results)

        if not documents:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "model": self.model,
                "documents_retrieved": 0,
                "cache_hit": False
            }

        # Step 2: Build context from retrieved documents
        context = self._build_context(documents)
        sources = self._extract_sources(documents)

        # Step 3: Generate answer using LLM
        logger.info("Generating answer with LLM...")
        llm_result = self._generate_answer(question, context)

        latency_ms = int((time.time() - start_time) * 1000)

        # Step 4: Estimate cost and log
        cost = self._estimate_cost(llm_result["input_tokens"], llm_result["output_tokens"])
        self._log_query(
            query_text=question,
            model_used=self.model,
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            cost=cost,
            latency_ms=latency_ms,
            cache_hit=cache_hit
        )

        return {
            "answer": llm_result["answer"],
            "sources": sources,
            "model": self.model,
            "documents_retrieved": len(documents),
            "cache_hit": cache_hit
        }

    def _build_context(self, documents: List[Dict]) -> str:
        """Build context from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            protocol = metadata.get('protocol', 'Unknown')
            doc_type = metadata.get('doc_type', 'document')

            context_parts.append(f"[Source {i} - {protocol} {doc_type}]")
            context_parts.append(doc['content'])
            context_parts.append("")

        return "\n".join(context_parts)

    def _extract_sources(self, documents: List[Dict]) -> List[Dict]:
        """Extract source information from documents"""
        sources = []

        for doc in documents:
            metadata = doc.get('metadata', {})
            sources.append({
                "protocol": metadata.get('protocol', 'Unknown'),
                "title": metadata.get('title', 'Unknown'),
                "doc_type": metadata.get('doc_type', 'document'),
                "url": metadata.get('url', ''),
                "similarity": doc.get('similarity', 0)
            })

        return sources

    def _generate_answer(self, question: str, context: str) -> Dict:
        """Generate answer using LLM with context. Returns answer text and token usage."""

        system_prompt = """You are a DeFi expert assistant. Answer questions about DeFi protocols
        based on the provided context from official whitepapers and documentation.
        Be accurate, concise, and cite which protocol's documentation you're referencing when relevant.
        If the context doesn't contain enough information to fully answer the question, say so."""

        user_prompt = f"""Context from DeFi documentation:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            usage = response.usage
            return {
                "answer": response.choices[0].message.content,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost based on model and token counts"""
        prices = MODEL_PRICING.get(self.model, {"input": 0.001, "output": 0.002})
        return (input_tokens / 1000 * prices["input"] +
                output_tokens / 1000 * prices["output"])

    def _log_query(self, query_text: str, model_used: str,
                   input_tokens: int, output_tokens: int,
                   cost: float, latency_ms: int, cache_hit: bool):
        """Log query to query_logs table for analytics"""
        conn = None
        cur = None
        try:
            conn = self.vector_store.get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO query_logs
                    (query_text, model_used, input_tokens, output_tokens, cost, latency_ms, cache_hit)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (query_text, model_used, input_tokens, output_tokens, cost, latency_ms, cache_hit))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
