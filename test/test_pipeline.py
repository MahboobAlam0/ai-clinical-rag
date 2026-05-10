"""
test/test_pipeline.py
---------------------
Unit tests for ingest chunking, retriever confidence logic, and RAG pipeline.

Run:
    pytest test/ -v --cov=src --cov-report=term-missing
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import get_settings
from src.ingest import chunk_text
from src.retriever import ClinicalRetriever, HIGH_CONF, MEDIUM_CONF, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure settings cache is reset between tests so env overrides take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        result = chunk_text("Short medical note.", size=400, overlap=80)
        assert len(result) == 1
        assert result[0] == "Short medical note."

    def test_long_text_produces_multiple_chunks(self) -> None:
        result = chunk_text("A" * 1000, size=400, overlap=80)
        assert len(result) > 1

    def test_overlap_creates_shared_content(self) -> None:
        chunks = chunk_text("word " * 200, size=200, overlap=50)
        assert len(chunks) >= 2

    def test_empty_chunks_filtered(self) -> None:
        result = chunk_text("   \n   ", size=400, overlap=80)
        assert result == []


# ── ClinicalRetriever ─────────────────────────────────────────────────────────

def _make_hit(score: float, idx: int) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "chunk": f"c{idx}", "pubid": f"pub{idx}",
        "question": "q", "label": "RESULTS", "decision": "yes",
    }
    return hit


def _make_retriever(scores: list[float]) -> ClinicalRetriever:
    """Build a retriever with its Qdrant client and encoder fully mocked."""
    retriever = ClinicalRetriever(top_k=len(scores))
    retriever._client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [_make_hit(s, i) for i, s in enumerate(scores)]
    retriever._client.query_points.return_value = mock_response
    retriever._model = MagicMock()
    retriever._model.encode.return_value = np.zeros((1, 4), dtype="float32")
    return retriever


class TestClinicalRetriever:
    def test_confidence_high(self) -> None:
        result = _make_retriever([0.85, 0.80, 0.75]).retrieve("query")
        assert result.confidence == "HIGH"

    def test_confidence_medium(self) -> None:
        result = _make_retriever([0.60, 0.55, 0.52]).retrieve("query")
        assert result.confidence == "MEDIUM"

    def test_confidence_low(self) -> None:
        result = _make_retriever([0.30, 0.25, 0.20]).retrieve("query")
        assert result.confidence == "LOW"

    def test_returns_correct_number_of_chunks(self) -> None:
        result = _make_retriever([0.80, 0.75, 0.70]).retrieve("query")
        assert len(result.chunks) == 3

    def test_scores_attached_to_chunks(self) -> None:
        result = _make_retriever([0.80, 0.70]).retrieve("query")
        assert result.chunks[0].score == 0.8
        assert result.chunks[1].score == 0.7

    def test_top_k_override_in_retrieve(self) -> None:
        """Per-request top_k overrides the instance default without mutation."""
        retriever = ClinicalRetriever(top_k=5)
        retriever._client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = [_make_hit(s, i) for i, s in enumerate([0.80, 0.75, 0.70])]
        retriever._client.query_points.return_value = mock_response
        retriever._model = MagicMock()
        retriever._model.encode.return_value = np.zeros((1, 4), dtype="float32")
        retriever.retrieve("query", top_k=3)
        retriever._client.query_points.assert_called_once()
        call_kwargs = retriever._client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 3


# ── ClinicalRAGPipeline ───────────────────────────────────────────────────────

class TestClinicalRAGPipeline:
    @patch("src.rag_pipeline.Groq")
    def test_answer_returns_rag_response(self, mock_groq_cls: MagicMock) -> None:
        from src.rag_pipeline import ClinicalRAGPipeline, RAGResponse

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mocked clinical answer."))]
        )

        pipeline = ClinicalRAGPipeline(top_k=2)
        pipeline.retriever._client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = [_make_hit(0.75, 0), _make_hit(0.65, 1)]
        pipeline.retriever._client.query_points.return_value = mock_response
        pipeline.retriever._model = MagicMock()
        pipeline.retriever._model.encode.return_value = np.zeros((1, 4), dtype="float32")

        os.environ["GROQ_API_KEY"] = "test_key_ci"
        result = pipeline.answer("What is the evidence for aspirin in stroke prevention?")

        assert isinstance(result, RAGResponse)
        assert result.answer == "Mocked clinical answer."
        assert result.confidence in ("HIGH", "MEDIUM", "LOW")
        assert len(result.sources) == 2

    @patch("src.rag_pipeline.Groq")
    def test_top_k_forwarded_to_retriever(self, mock_groq_cls: MagicMock) -> None:
        from src.rag_pipeline import ClinicalRAGPipeline

        mock_groq_cls.return_value = MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(
                        return_value=MagicMock(
                            choices=[MagicMock(message=MagicMock(content="answer"))]
                        )
                    )
                )
            )
        )

        pipeline = ClinicalRAGPipeline(top_k=5)
        pipeline.retriever._client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = [_make_hit(s, i) for i, s in enumerate([0.80, 0.75, 0.70])]
        pipeline.retriever._client.query_points.return_value = mock_response
        pipeline.retriever._model = MagicMock()
        pipeline.retriever._model.encode.return_value = np.zeros((1, 4), dtype="float32")

        os.environ["GROQ_API_KEY"] = "test_key_ci"
        pipeline.answer("query", top_k=3)

        call_kwargs = pipeline.retriever._client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 3
