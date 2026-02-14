"""
Search Optimization Module
Implements hybrid search with dynamic weighting, caching, and reranking
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Enhanced search result with multiple scoring dimensions"""
    id: int
    content: str
    metadata: Dict
    vector_score: float
    keyword_score: float
    hybrid_score: float
    rerank_score: Optional[float] = None
    latency_ms: Optional[int] = None


class HybridSearchOptimizer:
    """
    Hybrid search with dynamic weighting, caching, and performance tracking.
    Wraps a VectorStore to add adaptive weight selection, result combining,
    term-overlap reranking, and an LRU query cache.
    """

    def __init__(self, vector_store, cache_max_size: int = 1000, cache_ttl: int = 3600):
        self.vector_store = vector_store
        self.performance_history = defaultdict(list)
        self.optimal_weights = {'vector': 0.7, 'keyword': 0.3}
        self.query_cache = QueryCache(max_size=cache_max_size, ttl_seconds=cache_ttl)
        self._last_cache_hit = False

    def adaptive_search(
        self,
        query: str,
        top_k: int = 10,
        auto_adjust_weights: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search with adaptive weight optimization and caching.
        """
        start_time = time.time()

        # Check cache first
        cache_key = f"{query}::{top_k}"
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            self._last_cache_hit = True
            return cached

        self._last_cache_hit = False

        # Get current optimal weights
        if auto_adjust_weights:
            weights = self._get_optimal_weights(query)
        else:
            weights = self.optimal_weights

        # Perform both searches against real vector store
        vector_results = self._vector_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)

        # Combine with weighted scoring
        combined_results = self._combine_results(
            vector_results,
            keyword_results,
            weights['vector'],
            weights['keyword']
        )

        # Rerank using term overlap boost
        reranked_results = self._rerank_results(combined_results, query)

        # Select top-k
        final_results = reranked_results[:top_k]

        # Track performance
        latency_ms = int((time.time() - start_time) * 1000)
        self._track_performance(query, final_results, latency_ms)

        # Cache results
        self.query_cache.set(cache_key, final_results)

        return final_results

    def was_cache_hit(self) -> bool:
        """Check if the most recent query was served from cache"""
        return self._last_cache_hit

    def search_results_to_dicts(self, results: List[SearchResult]) -> List[Dict]:
        """Convert SearchResult objects to dicts compatible with RAGEngine"""
        return [
            {
                'id': r.id,
                'content': r.content,
                'metadata': r.metadata,
                'similarity': r.rerank_score if r.rerank_score is not None else r.hybrid_score
            }
            for r in results
        ]

    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """Perform vector similarity search via vector_store"""
        try:
            results = self.vector_store.search(query, top_k=limit)
            return [
                {
                    'id': r['id'],
                    'content': r['content'],
                    'metadata': r['metadata'],
                    'score': float(r.get('similarity', 0))
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Perform keyword/full-text search via vector_store"""
        try:
            results = self.vector_store.keyword_search(query, top_k=limit)
            return [
                {
                    'id': r['id'],
                    'content': r['content'],
                    'metadata': r['metadata'],
                    'score': float(r.get('score', 0))
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        vector_weight: float,
        keyword_weight: float
    ) -> List[SearchResult]:
        """
        Combine vector and keyword results with weighted scoring
        """
        combined = {}

        # Process vector results
        for result in vector_results:
            doc_id = result['id']
            combined[doc_id] = SearchResult(
                id=doc_id,
                content=result['content'],
                metadata=result['metadata'],
                vector_score=result['score'],
                keyword_score=0,
                hybrid_score=result['score'] * vector_weight,
                latency_ms=None
            )

        # Process keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined:
                # Document appears in both - update scores
                combined[doc_id].keyword_score = result['score']
                combined[doc_id].hybrid_score = (
                    combined[doc_id].vector_score * vector_weight +
                    result['score'] * keyword_weight
                )
            else:
                # Document only in keyword results
                combined[doc_id] = SearchResult(
                    id=doc_id,
                    content=result['content'],
                    metadata=result['metadata'],
                    vector_score=0,
                    keyword_score=result['score'],
                    hybrid_score=result['score'] * keyword_weight,
                    latency_ms=None
                )

        # Sort by hybrid score
        results = sorted(combined.values(), key=lambda x: x.hybrid_score, reverse=True)
        return results

    def _rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rerank results using term overlap boost.
        Blends the hybrid score (70%) with a term-overlap relevance signal (30%).
        """
        for result in results:
            overlap_score = self._compute_term_overlap_boost(query, result.content)
            result.rerank_score = (result.hybrid_score * 0.7 + overlap_score * 0.3)

        reranked = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        return reranked

    def _compute_term_overlap_boost(self, query: str, document: str) -> float:
        """
        Compute a lightweight relevance boost based on query-document term overlap.
        Uses set intersection of lowercased terms, boosted by 1.5x and clamped to [0, 1].
        """
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms & doc_terms) / len(query_terms)
        return min(overlap * 1.5, 1.0)

    def _get_optimal_weights(self, query: str) -> Dict[str, float]:
        """
        Determine optimal weights based on query characteristics.
        Short queries favour keyword search; longer queries favour vector search.
        """
        query_length = len(query.split())

        if query_length <= 3:
            return {'vector': 0.5, 'keyword': 0.5}
        elif query_length <= 7:
            return {'vector': 0.6, 'keyword': 0.4}
        else:
            return {'vector': 0.75, 'keyword': 0.25}

    def _track_performance(self, query: str, results: List[SearchResult], latency_ms: int):
        """Track search performance for optimization"""
        self.performance_history[query[:50]].append({
            'timestamp': time.time(),
            'latency_ms': latency_ms,
            'num_results': len(results),
            'avg_score': float(np.mean([r.hybrid_score for r in results])) if results else 0
        })

    def optimize_weights_from_feedback(
        self,
        query_feedback: List[Tuple[str, int, bool]]
    ) -> Dict[str, float]:
        """
        Optimize weights based on user feedback.
        query_feedback: List of (query, result_position, was_relevant)
        """
        vector_success = []
        keyword_success = []

        for query, position, relevant in query_feedback:
            results = self.adaptive_search(query, top_k=position+1, auto_adjust_weights=False)
            if position < len(results):
                result = results[position]
                if relevant:
                    vector_success.append(result.vector_score)
                    keyword_success.append(result.keyword_score)

        if vector_success and keyword_success:
            avg_vector = np.mean(vector_success)
            avg_keyword = np.mean(keyword_success)
            total = avg_vector + avg_keyword

            if total > 0:
                self.optimal_weights = {
                    'vector': float(avg_vector / total),
                    'keyword': float(avg_keyword / total)
                }

        return self.optimal_weights

    def generate_performance_report(self) -> Dict:
        """Generate performance optimization report"""
        report = {
            'total_queries': len(self.performance_history),
            'avg_latency_ms': 0,
            'p95_latency_ms': 0,
            'optimal_weights': self.optimal_weights,
            'cache_stats': self.query_cache.get_stats(),
            'query_patterns': []
        }

        all_latencies = []
        for query, history in self.performance_history.items():
            latencies = [h['latency_ms'] for h in history]
            all_latencies.extend(latencies)

            report['query_patterns'].append({
                'query_prefix': query,
                'count': len(history),
                'avg_latency': float(np.mean(latencies)) if latencies else 0,
                'avg_score': float(np.mean([h['avg_score'] for h in history]))
            })

        if all_latencies:
            report['avg_latency_ms'] = float(np.mean(all_latencies))
            report['p95_latency_ms'] = float(np.percentile(all_latencies, 95))

        return report


class QueryCache:
    """
    LRU cache for query results to reduce costs and latency
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[list]:
        """Get cached result if available and not expired"""
        if query in self.cache:
            if time.time() - self.access_times[query] < self.ttl_seconds:
                self.hits += 1
                self.access_times[query] = time.time()
                return self.cache[query]
            else:
                del self.cache[query]
                del self.access_times[query]

        self.misses += 1
        return None

    def set(self, query: str, result: list):
        """Cache a query result"""
        if len(self.cache) >= self.max_size:
            lru_query = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_query]
            del self.access_times[lru_query]

        self.cache[query] = result
        self.access_times[query] = time.time()

    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(self.hits / total, 4) if total > 0 else 0,
        }
