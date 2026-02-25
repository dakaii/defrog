"""
Tests for API security: authentication, CORS, rate limiting, and input validation.

Uses FastAPI's TestClient with module-level mocks so no real DB or LLM is needed.
"""
import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.append('/app')

# Ensure OPENAI_API_KEY is set before importing src.main (needed by VectorStore init)
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-tests")

# ---------------------------------------------------------------------------
# Module-level mocks — must be set up BEFORE src.main is imported, because
# RAGEngine() and VectorStore() are called at module level in main.py.
# ---------------------------------------------------------------------------
_MOCK_QUERY_RESULT = {
    "answer": "Uniswap V3 uses concentrated liquidity.",
    "sources": [],
    "model": "gpt-4o-mini",
    "documents_retrieved": 3,
    "cache_hit": False,
}

_mock_engine = Mock()
_mock_engine.query.return_value = _MOCK_QUERY_RESULT
_mock_engine.optimizer.query_cache.get_stats.return_value = {"hits": 0, "misses": 0}

_mock_store = Mock()
_mock_store.get_document_count.return_value = 42
_mock_store.get_connection.return_value = Mock()

# Remove any cached version so our patches take effect on a clean import
for _key in list(sys.modules.keys()):
    if "src.main" in _key:
        del sys.modules[_key]

with patch("src.retrieval.rag_engine.RAGEngine", return_value=_mock_engine), \
     patch("src.ingestion.vector_store.VectorStore", return_value=_mock_store):
    import src.main as _main
    from src.main import app

# Hold a reference to the limiter object captured by the @limiter.limit() decorators.
# We must reset THIS object's storage — replacing _main.limiter has no effect because
# the decorators already closed over the original instance.
_original_limiter = _main.limiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear the rate limiter storage before (and after) every test so counters
    don't bleed across tests."""
    _original_limiter._storage.reset()
    yield
    _original_limiter._storage.reset()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


_VALID_QUERY = {"query": "How does Uniswap V3 concentrated liquidity work?"}


# ===========================================================================
# Authentication
# ===========================================================================

class TestAPIKeyAuth:
    def test_no_api_key_configured_allows_all(self, client, monkeypatch):
        """When API_KEY is empty, all requests pass through (dev mode)."""
        monkeypatch.setattr(_main, "_API_KEY", "")
        resp = client.post("/query", json=_VALID_QUERY)
        assert resp.status_code == 200

    def test_correct_key_is_accepted(self, client, monkeypatch):
        """Correct X-API-Key header passes auth."""
        monkeypatch.setattr(_main, "_API_KEY", "test-secret")
        resp = client.post(
            "/query",
            json=_VALID_QUERY,
            headers={"X-API-Key": "test-secret"},
        )
        assert resp.status_code == 200

    def test_wrong_key_returns_401(self, client, monkeypatch):
        """Wrong key value returns 401."""
        monkeypatch.setattr(_main, "_API_KEY", "test-secret")
        resp = client.post(
            "/query",
            json=_VALID_QUERY,
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_missing_key_returns_401(self, client, monkeypatch):
        """Omitting X-API-Key header returns 401 when auth is enabled."""
        monkeypatch.setattr(_main, "_API_KEY", "test-secret")
        resp = client.post("/query", json=_VALID_QUERY)
        assert resp.status_code == 401

    def test_health_is_always_public(self, client, monkeypatch):
        """/health never requires an API key."""
        monkeypatch.setattr(_main, "_API_KEY", "test-secret")
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_protocols_is_always_public(self, client, monkeypatch):
        """/protocols never requires an API key."""
        monkeypatch.setattr(_main, "_API_KEY", "test-secret")
        resp = client.get("/protocols")
        assert resp.status_code == 200

    def test_analytics_endpoints_are_public(self, client, monkeypatch):
        """/analytics/* endpoints are read-only and never require a key."""
        monkeypatch.setattr(_main, "_API_KEY", "test-secret")
        resp = client.get("/analytics/cache")
        assert resp.status_code == 200


# ===========================================================================
# Input Validation
# ===========================================================================

class TestInputValidation:
    def test_empty_query_returns_422(self, client, monkeypatch):
        monkeypatch.setattr(_main, "_API_KEY", "")
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_whitespace_only_query_returns_422(self, client, monkeypatch):
        monkeypatch.setattr(_main, "_API_KEY", "")
        resp = client.post("/query", json={"query": "   "})
        assert resp.status_code == 422

    def test_query_over_1000_chars_returns_422(self, client, monkeypatch):
        monkeypatch.setattr(_main, "_API_KEY", "")
        resp = client.post("/query", json={"query": "a" * 1001})
        assert resp.status_code == 422
        assert "1000" in resp.json()["detail"]

    def test_query_at_exactly_1000_chars_is_accepted(self, client, monkeypatch):
        monkeypatch.setattr(_main, "_API_KEY", "")
        resp = client.post("/query", json={"query": "a" * 1000})
        assert resp.status_code == 200

    def test_valid_query_returns_expected_response_shape(self, client, monkeypatch):
        monkeypatch.setattr(_main, "_API_KEY", "")
        resp = client.post("/query", json=_VALID_QUERY)
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "model_used" in body
        assert "latency_ms" in body
        assert "documents_retrieved" in body


# ===========================================================================
# Rate Limiting
# ===========================================================================

class TestRateLimiting:
    def test_query_allows_10_requests_then_429(self, client, monkeypatch):
        """The 11th POST /query within a minute returns 429."""
        monkeypatch.setattr(_main, "_API_KEY", "")
        for i in range(10):
            resp = client.post("/query", json=_VALID_QUERY)
            assert resp.status_code == 200, f"Request {i + 1} unexpectedly failed: {resp.json()}"
        resp = client.post("/query", json=_VALID_QUERY)
        assert resp.status_code == 429

    def test_ingest_allows_2_requests_then_429(self, client, monkeypatch):
        """The 3rd POST /ingest within a minute returns 429."""
        monkeypatch.setattr(_main, "_API_KEY", "")
        # Stub out the background crawler so it returns instantly and the
        # 1-minute rate-limit window doesn't expire before the 3rd request.
        monkeypatch.setattr(_main, "_run_ingestion", lambda *a, **kw: None)
        for _ in range(2):
            resp = client.post("/ingest", json={})
            assert resp.status_code == 200
        resp = client.post("/ingest", json={})
        assert resp.status_code == 429


# ===========================================================================
# CORS
# ===========================================================================

class TestCORS:
    def test_allowed_origin_receives_cors_header(self, monkeypatch):
        """Preflight from an allowed origin gets the Access-Control-Allow-Origin header."""
        monkeypatch.setattr(_main, "_API_KEY", "")
        from fastapi.testclient import TestClient
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.options(
            "/query",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"

    def test_disallowed_origin_does_not_receive_cors_header(self, monkeypatch):
        """Preflight from an unknown origin does not echo back that origin."""
        monkeypatch.setattr(_main, "_API_KEY", "")
        from fastapi.testclient import TestClient
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.options(
            "/query",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "https://evil.example.com"
