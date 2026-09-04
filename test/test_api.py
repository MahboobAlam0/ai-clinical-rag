"""
test/test_api.py
----------------
Integration tests for the FastAPI service using httpx TestClient.
All heavy dependencies (FAISS, LLM) are mocked so tests run in CI
without GPU or a real Groq key.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.rag_pipeline import RAGResponse


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mock_pipeline() -> MagicMock:
    """Return a pipeline mock that produces a deterministic RAGResponse."""
    mock = MagicMock()
    mock.retriever._index.ntotal = 5000
    mock.answer.return_value = RAGResponse(
        query="test question",
        answer="Mocked clinical answer.",
        confidence="HIGH",
        confidence_note="Strong evidence found in retrieved literature.",
        max_score=0.85,
        mean_score=0.78,
        sources=[
            {
                "index": 1,
                "pubid": "12345",
                "label": "RESULTS",
                "score": 0.85,
                "excerpt": "Relevant clinical excerpt...",
            }
        ],
    )
    return mock


@pytest.fixture()
def client():
    """TestClient with pipeline fully mocked out."""
    from api.main import app, get_pipeline, _build_pipeline

    mock_pipeline = _make_mock_pipeline()

    # Override both the DI function and the cached singleton
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    _build_pipeline.cache_clear()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()
    _build_pipeline.cache_clear()


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "index_size" in data
        assert "model" in data
        assert "uptime_seconds" in data


# ── Metrics endpoint ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_schema(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        for field in ("total_queries", "cache_hits", "confidence_distribution", "avg_latency_ms"):
            assert field in data


# ── Query endpoint ────────────────────────────────────────────────────────────

class TestQuery:
    def test_valid_query_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What is the evidence for metformin in type 2 diabetes?", "top_k": 5},
        )
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What is the evidence for metformin in type 2 diabetes?"},
        )
        data = resp.json()
        for field in (
            "question", "answer", "confidence", "confidence_note",
            "max_score", "mean_score", "sources", "latency_ms", "cached", "request_id",
        ):
            assert field in data, f"Missing field: {field}"

    def test_request_id_in_response_header(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What is the evidence for statins?"},
        )
        assert "x-request-id" in resp.headers

    def test_question_too_short_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_question_too_long_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "x" * 501})
        assert resp.status_code == 422

    def test_top_k_out_of_range_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "Valid question about diabetes?", "top_k": 99},
        )
        assert resp.status_code == 422

    def test_second_identical_query_is_cached(self, client: TestClient) -> None:
        payload = {"question": "What is the evidence for aspirin in stroke?", "top_k": 5}
        client.post("/query", json=payload)              # first — cache miss
        resp = client.post("/query", json=payload)       # second — cache hit
        assert resp.json()["cached"] is True

    def test_pipeline_answer_called_with_correct_top_k(self, client: TestClient) -> None:
        from api.main import get_pipeline
        mock_pipeline = client.app.dependency_overrides[get_pipeline]()
        mock_pipeline.answer.reset_mock()

        client.post(
            "/query",
            json={"question": "What is the evidence for beta-blockers?", "top_k": 3},
        )
        mock_pipeline.answer.assert_called_once_with(
            "What is the evidence for beta-blockers?", top_k=3
        )


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_no_auth_required_when_api_key_unset(self, client: TestClient) -> None:
        """Default settings have API_KEY='', so auth is disabled."""
        resp = client.post(
            "/query",
            json={"question": "What is the evidence for statins?"},
        )
        assert resp.status_code == 200

    def test_wrong_api_key_returns_401(self, client: TestClient) -> None:
        from src.config import get_settings
        settings = get_settings()
        original = settings.api_key
        try:
            object.__setattr__(settings, "api_key", "secret-key")
            resp = client.post(
                "/query",
                json={"question": "Valid question about diabetes?"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 401
        finally:
            object.__setattr__(settings, "api_key", original)
