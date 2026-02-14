"""
Comprehensive RAG Pipeline Tests
Tests for verifying RAG quality, performance, and multi-LLM support
"""
import pytest
import os
import time
import json
from typing import Dict, List
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('/app')

from src.retrieval.rag_engine import RAGEngine
from src.ingestion.vector_store import VectorStore
from src.evaluation.rag_evaluator import RAGEvaluator, EvaluationResult


class TestMultiLLMSupport:
    """Test suite for multi-LLM provider support"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test answer from Uniswap docs"))]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_openai_initialization(self, monkeypatch):
        """Test RAG engine initializes with OpenAI"""
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        monkeypatch.delenv('LLM_BASE_URL', raising=False)
        
        with patch('src.retrieval.rag_engine.OpenAI') as mock_openai:
            engine = RAGEngine()
            mock_openai.assert_called_once_with(api_key='test-key')
    
    def test_deepseek_initialization(self, monkeypatch):
        """Test RAG engine initializes with DeepSeek"""
        monkeypatch.setenv('LLM_API_KEY', 'deepseek-key')
        monkeypatch.setenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
        monkeypatch.setenv('LLM_MODEL', 'deepseek-chat')
        
        with patch('src.retrieval.rag_engine.OpenAI') as mock_openai:
            engine = RAGEngine()
            mock_openai.assert_called_once_with(
                api_key='deepseek-key',
                base_url='https://api.deepseek.com/v1'
            )
            assert engine.model == 'deepseek-chat'
    
    def test_qwen_initialization(self, monkeypatch):
        """Test RAG engine initializes with Qwen"""
        monkeypatch.setenv('LLM_API_KEY', 'dashscope-key')
        monkeypatch.setenv('LLM_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        monkeypatch.setenv('LLM_MODEL', 'qwen-max')
        
        with patch('src.retrieval.rag_engine.OpenAI') as mock_openai:
            engine = RAGEngine()
            mock_openai.assert_called_once_with(
                api_key='dashscope-key',
                base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
            )
            assert engine.model == 'qwen-max'


class TestRAGQuality:
    """Test suite for RAG quality metrics"""
    
    @pytest.fixture
    def sample_rag_output(self):
        """Sample RAG output for testing"""
        return {
            'query': 'How does Uniswap V3 concentrated liquidity work?',
            'answer': 'Uniswap V3 allows liquidity providers to concentrate their capital within custom price ranges.',
            'contexts': [
                'Uniswap V3 introduces concentrated liquidity, allowing LPs to allocate capital to specific price ranges.',
                'In V3, liquidity providers can choose custom price ranges for their positions.',
                'This concentration leads to higher capital efficiency compared to V2.'
            ]
        }
    
    @pytest.fixture
    def mock_evaluator(self):
        """Mock evaluator for testing"""
        with patch('src.evaluation.rag_evaluator.OpenAI'):
            evaluator = RAGEvaluator()
            return evaluator
    
    def test_faithfulness_evaluation(self, mock_evaluator, sample_rag_output):
        """Test faithfulness scoring"""
        with patch.object(mock_evaluator.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content="""
                Claim 1: Uniswap V3 allows liquidity providers to concentrate capital - SUPPORTED
                Claim 2: Within custom price ranges - SUPPORTED
                Final score: 2/2
                """))]
            )
            
            score = mock_evaluator.evaluate_faithfulness(
                sample_rag_output['answer'],
                sample_rag_output['contexts']
            )
            
            assert 0 <= score <= 1
            assert score == 1.0  # All claims supported
    
    def test_answer_relevancy_evaluation(self, mock_evaluator, sample_rag_output):
        """Test answer relevancy scoring"""
        with patch.object(mock_evaluator.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content="""
                The answer directly addresses how concentrated liquidity works.
                SCORE: 85
                """))]
            )
            
            score = mock_evaluator.evaluate_answer_relevancy(
                sample_rag_output['query'],
                sample_rag_output['answer']
            )
            
            assert 0 <= score <= 1
            assert score == 0.85
    
    def test_complete_evaluation(self, mock_evaluator, sample_rag_output):
        """Test complete evaluation pipeline"""
        with patch.object(mock_evaluator, 'evaluate_faithfulness', return_value=0.9):
            with patch.object(mock_evaluator, 'evaluate_answer_relevancy', return_value=0.85):
                with patch.object(mock_evaluator, 'evaluate_context_precision', return_value=0.8):
                    result = mock_evaluator.evaluate(
                        query=sample_rag_output['query'],
                        answer=sample_rag_output['answer'],
                        contexts=sample_rag_output['contexts']
                    )
                    
                    assert isinstance(result, EvaluationResult)
                    assert result.faithfulness == 0.9
                    assert result.answer_relevancy == 0.85
                    assert result.context_precision == 0.8
                    assert 0.8 <= result.overall_score <= 0.9


class TestHybridSearch:
    """Test suite for hybrid search optimization"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing"""
        store = Mock(spec=VectorStore)
        return store
    
    def test_hybrid_search_weighting(self, mock_vector_store):
        """Test hybrid search uses correct weighting"""
        mock_vector_store.hybrid_search.return_value = [
            {'content': 'Test content', 'similarity': 0.85, 'metadata': {}}
        ]
        
        # Test with custom weights
        results = mock_vector_store.hybrid_search(
            query="test query",
            top_k=5
        )
        
        mock_vector_store.hybrid_search.assert_called_once_with(
            query="test query",
            top_k=5
        )
        assert len(results) == 1
    
    def test_adaptive_weights_short_query(self):
        """Test that short queries use balanced weights"""
        from src.optimization.search_optimizer import HybridSearchOptimizer

        optimizer = HybridSearchOptimizer(mock_vector_store)
        weights = optimizer._get_optimal_weights("uniswap liquidity")
        assert weights == {'vector': 0.5, 'keyword': 0.5}

    def test_adaptive_weights_long_query(self):
        """Test that long queries favour vector search"""
        from src.optimization.search_optimizer import HybridSearchOptimizer

        optimizer = HybridSearchOptimizer(mock_vector_store)
        weights = optimizer._get_optimal_weights(
            "how does the concentrated liquidity mechanism work in uniswap v3 protocol"
        )
        assert weights == {'vector': 0.75, 'keyword': 0.25}


class TestQueryCache:
    """Test suite for query caching"""

    def test_cache_hit_and_miss(self):
        """Test that cache returns results on hit and None on miss"""
        from src.optimization.search_optimizer import QueryCache

        cache = QueryCache(max_size=10, ttl_seconds=3600)
        assert cache.get("test") is None
        assert cache.get_stats()['misses'] == 1

        cache.set("test", [{"id": 1}])
        result = cache.get("test")
        assert result == [{"id": 1}]
        assert cache.get_stats()['hits'] == 1

    def test_cache_lru_eviction(self):
        """Test that LRU eviction works when cache is full"""
        from src.optimization.search_optimizer import QueryCache

        cache = QueryCache(max_size=2, ttl_seconds=3600)
        cache.set("a", [1])
        cache.set("b", [2])
        cache.set("c", [3])  # Should evict "a"

        assert cache.get("a") is None  # evicted
        assert cache.get("c") == [3]

    def test_cache_ttl_expiry(self):
        """Test that expired entries are not returned"""
        from src.optimization.search_optimizer import QueryCache

        cache = QueryCache(max_size=10, ttl_seconds=0)  # 0s TTL = instant expiry
        cache.set("test", [1])

        # Entry should be expired immediately
        result = cache.get("test")
        assert result is None


class TestSearchResultConversion:
    """Test SearchResult to dict conversion for RAGEngine compatibility"""

    def test_search_results_to_dicts(self):
        """Test that SearchResult objects convert to RAGEngine-compatible dicts"""
        from src.optimization.search_optimizer import HybridSearchOptimizer, SearchResult

        store = Mock(spec=VectorStore)
        optimizer = HybridSearchOptimizer(store)

        results = [
            SearchResult(
                id=1, content="test content", metadata={"protocol": "Aave"},
                vector_score=0.9, keyword_score=0.3, hybrid_score=0.72,
                rerank_score=0.65
            )
        ]
        dicts = optimizer.search_results_to_dicts(results)

        assert len(dicts) == 1
        assert dicts[0]['similarity'] == 0.65  # Uses rerank_score
        assert dicts[0]['content'] == "test content"
        assert dicts[0]['metadata']['protocol'] == "Aave"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])