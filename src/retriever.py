"""
src/retriever.py
----------------
Qdrant-backed retriever that returns top-k chunks with
cosine similarity scores used for uncertainty estimation.

Local mode  (default): QdrantClient(path="qdrant_storage") — no server needed.
Remote mode (Docker) : set QDRANT_HOST=localhost in .env
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)

# Module-level aliases kept for backwards-compatibility with tests
HIGH_CONF = 0.70
MEDIUM_CONF = 0.50


@dataclass
class RetrievedChunk:
    chunk: str
    pubid: str
    question: str
    label: str
    score: float


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    max_score: float
    mean_score: float
    confidence: str
    confidence_note: str


def _build_qdrant_client(settings) -> QdrantClient:
    mode = settings.qdrant_mode.lower()
    if mode == "memory":
        logger.info("Using in-memory Qdrant (ephemeral)")
        return QdrantClient(":memory:")
    if mode == "remote" or settings.qdrant_host:
        logger.info("Connecting to Qdrant server", extra={"host": settings.qdrant_host, "port": settings.qdrant_port})
        return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    logger.info("Using local Qdrant storage", extra={"path": settings.qdrant_path})
    return QdrantClient(path=settings.qdrant_path)


class ClinicalRetriever:
    def __init__(self, top_k: int | None = None) -> None:
        settings = get_settings()
        self._default_top_k: int = top_k if top_k is not None else settings.top_k_default
        self._model: SentenceTransformer | None = None
        self._client: QdrantClient | None = None

    def _load(self) -> None:
        if self._client is not None:
            return
        settings = get_settings()
        self._model = SentenceTransformer(settings.embedding_model)
        self._client = _build_qdrant_client(settings)
        count = self._client.count(settings.qdrant_collection).count
        logger.info("Qdrant collection ready", extra={"vectors": count})

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Return top-k chunks with confidence classification.

        top_k overrides the instance default per-request — thread-safe,
        no shared mutable state.
        """
        self._load()
        settings = get_settings()
        k = top_k if top_k is not None else self._default_top_k

        q_vec = self._model.encode([query], normalize_embeddings=True)[0].tolist()

        response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=q_vec,
            limit=k,
            with_payload=True,
        )
        hits = response.points

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(
                chunk=hit.payload["chunk"],
                pubid=hit.payload["pubid"],
                question=hit.payload["question"],
                label=hit.payload["label"],
                score=round(float(hit.score), 4),
            )
            for hit in hits
        ]

        scores = [c.score for c in chunks]
        max_score = max(scores) if scores else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0

        high = settings.confidence_high
        medium = settings.confidence_medium

        if max_score >= high:
            confidence = "HIGH"
            confidence_note = "Strong evidence found in retrieved literature."
        elif max_score >= medium:
            confidence = "MEDIUM"
            confidence_note = "Partial evidence found. Verify with a clinician."
        else:
            confidence = "LOW"
            confidence_note = (
                "Weak retrieval signal. Answer may be unreliable — "
                "consult primary literature or a clinician."
            )

        logger.info(
            "Retrieval complete",
            extra={"top_k": k, "max_score": max_score, "confidence": confidence},
        )

        return RetrievalResult(
            chunks=chunks,
            max_score=round(max_score, 4),
            mean_score=round(mean_score, 4),
            confidence=confidence,
            confidence_note=confidence_note,
        )
