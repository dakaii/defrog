"""
Search Optimization Module
Implements advanced hybrid search with dynamic weighting and reranking
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from collections import defaultdict
import json

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
    Advanced hybrid search with dynamic weighting and performance optimization
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.performance_history = defaultdict(list)
        self.optimal_weights = {'vector': 0.7, 'keyword': 0.3}
        
    def adaptive_search(
        self, 
        query: str, 
        top_k: int = 10,
        auto_adjust_weights: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search with adaptive weight optimization
        """
        start_time = time.time()
        
        # Get current optimal weights
        if auto_adjust_weights:
            weights = self._get_optimal_weights(query)
        else:
            weights = self.optimal_weights
        
        # Perform searches in parallel (simulated)
        vector_results = self._vector_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and rerank
        combined_results = self._combine_results(
            vector_results, 
            keyword_results,
            weights['vector'],
            weights['keyword']
        )
        
        # Rerank using cross-encoder or additional signals
        reranked_results = self._rerank_results(combined_results, query)
        
        # Select top-k
        final_results = reranked_results[:top_k]
        
        # Track performance
        latency_ms = int((time.time() - start_time) * 1000)
        self._track_performance(query, final_results, latency_ms)
        
        return final_results
    
    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """Perform vector similarity search"""
        # This would call the actual vector store
        # Simulated for testing
        results = []
        for i in range(min(limit, 5)):
            results.append({
                'id': i,
                'content': f"Vector result {i} for query: {query}",
                'metadata': {'source': 'vector'},
                'score': 0.9 - (i * 0.1)
            })
        return results
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Perform keyword/BM25 search"""
        # This would use PostgreSQL full-text search
        # Simulated for testing
        results = []
        for i in range(min(limit, 5)):
            results.append({
                'id': i + 100,  # Different ID space
                'content': f"Keyword result {i} for query: {query}",
                'metadata': {'source': 'keyword'},
                'score': 0.85 - (i * 0.15)
            })
        return results
    
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
        Rerank results using additional signals or cross-encoder
        """
        # Factors for reranking:
        # 1. Query-document similarity (cross-encoder)
        # 2. Document freshness
        # 3. Source authority
        # 4. Length penalty
        
        for result in results:
            # Simulate cross-encoder scoring
            cross_score = self._compute_cross_encoder_score(query, result.content)
            
            # Combine with original score
            result.rerank_score = (result.hybrid_score * 0.7 + cross_score * 0.3)
        
        # Re-sort by rerank score
        reranked = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        return reranked
    
    def _compute_cross_encoder_score(self, query: str, document: str) -> float:
        """
        Compute cross-encoder score for query-document pair
        In production, this would use a model like MS MARCO
        """
        # Simplified scoring based on overlap
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & doc_terms) / len(query_terms)
        return min(overlap * 1.5, 1.0)  # Boost overlap score
    
    def _get_optimal_weights(self, query: str) -> Dict[str, float]:
        """
        Determine optimal weights based on query characteristics
        """
        # Analyze query to determine best weight distribution
        query_length = len(query.split())
        
        # Heuristics:
        # - Short queries: favor keyword search
        # - Long queries: favor vector search
        # - Technical queries: balanced
        
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
            'avg_score': np.mean([r.hybrid_score for r in results]) if results else 0
        })
    
    def optimize_weights_from_feedback(
        self,
        query_feedback: List[Tuple[str, int, bool]]
    ) -> Dict[str, float]:
        """
        Optimize weights based on user feedback
        query_feedback: List of (query, result_position, was_relevant)
        """
        # Collect feedback statistics
        vector_success = []
        keyword_success = []
        
        for query, position, relevant in query_feedback:
            # Get the result at position
            results = self.adaptive_search(query, top_k=position+1, auto_adjust_weights=False)
            if position < len(results):
                result = results[position]
                if relevant:
                    vector_success.append(result.vector_score)
                    keyword_success.append(result.keyword_score)
        
        # Adjust weights based on success rates
        if vector_success and keyword_success:
            avg_vector = np.mean(vector_success)
            avg_keyword = np.mean(keyword_success)
            total = avg_vector + avg_keyword
            
            if total > 0:
                self.optimal_weights = {
                    'vector': avg_vector / total,
                    'keyword': avg_keyword / total
                }
        
        return self.optimal_weights
    
    def generate_performance_report(self) -> Dict:
        """Generate performance optimization report"""
        report = {
            'total_queries': len(self.performance_history),
            'avg_latency_ms': 0,
            'p95_latency_ms': 0,
            'optimal_weights': self.optimal_weights,
            'query_patterns': []
        }
        
        all_latencies = []
        for query, history in self.performance_history.items():
            latencies = [h['latency_ms'] for h in history]
            all_latencies.extend(latencies)
            
            report['query_patterns'].append({
                'query_prefix': query,
                'count': len(history),
                'avg_latency': np.mean(latencies) if latencies else 0,
                'avg_score': np.mean([h['avg_score'] for h in history])
            })
        
        if all_latencies:
            report['avg_latency_ms'] = np.mean(all_latencies)
            report['p95_latency_ms'] = np.percentile(all_latencies, 95)
        
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
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached result if available and not expired"""
        if query in self.cache:
            # Check if expired
            if time.time() - self.access_times[query] < self.ttl_seconds:
                self.hits += 1
                self.access_times[query] = time.time()  # Update access time
                return self.cache[query]
            else:
                # Expired - remove
                del self.cache[query]
                del self.access_times[query]
        
        self.misses += 1
        return None
    
    def set(self, query: str, result: Dict):
        """Cache a query result"""
        # Implement LRU eviction if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used
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
            'hit_rate': self.hits / total if total > 0 else 0,
            'cost_savings': self.hits * 0.002  # Approximate $ saved per cached query
        }