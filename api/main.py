"""
api/main.py
-----------
FastAPI service for the Clinical RAG pipeline.

Endpoints:
  POST /query      — answer a clinical question
  GET  /health     — liveness + readiness check
  GET  /metrics    — aggregate query statistics

Run locally:
  uvicorn api.main:app --reload --port 8000

Docker:
  docker-compose up
"""

import json
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import Settings, get_settings
from src.logger import get_logger
from src.rag_pipeline import ClinicalRAGPipeline, RAGResponse

logger = get_logger(__name__)


# ── In-process LRU cache (query + top_k → RAGResponse) ──────────────────────

class _LRUCache:
    """Thread-safe LRU cache using OrderedDict. Avoids re-calling the LLM for
    identical queries, cutting latency and Groq quota usage."""

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, RAGResponse] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> RAGResponse | None:
        if key not in self._cache:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def set(self, key: str, value: RAGResponse) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


_cache = _LRUCache()


# ── In-memory metrics ────────────────────────────────────────────────────────

@dataclass
class _Metrics:
    total_queries: int = 0
    cache_hits: int = 0
    confidence_counts: dict[str, int] = field(
        default_factory=lambda: {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    )
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)


_metrics = _Metrics()


# ── Pipeline singleton (dependency-injectable) ────────────────────────────────

@lru_cache
def _build_pipeline() -> ClinicalRAGPipeline:
    pipeline = ClinicalRAGPipeline()
    pipeline.retriever._load()
    logger.info("Pipeline warmed up")
    return pipeline


def get_pipeline() -> ClinicalRAGPipeline:
    return _build_pipeline()


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_pipeline()   # warm up at startup so first request is fast
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical RAG API",
    description=(
        "Retrieval-Augmented Generation over PubMed literature "
        "with uncertainty quantification."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Middlewares ───────────────────────────────────────────────────────────────

class _RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(_RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """No-op when API_KEY is unset (local / dev). Enforced in production."""
    if settings.api_key and key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        examples=["What are the effects of metformin on HbA1c in type 2 diabetes?"],
    )
    top_k: int = Field(default=5, ge=1, le=10)


class SourceDoc(BaseModel):
    index: int
    pubid: str
    label: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    confidence_note: str
    max_score: float
    mean_score: float
    sources: list[SourceDoc]
    latency_ms: float
    cached: bool
    request_id: str


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_queries: int
    cache_hits: int
    confidence_distribution: dict[str, int]
    avg_latency_ms: float
    cache_stats: dict[str, int]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health(pipeline: ClinicalRAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    try:
        settings = get_settings()
        count = pipeline.retriever._client.count(settings.qdrant_collection).count
    except Exception:
        count = 0
    return HealthResponse(
        status="ok",
        index_size=count,
        model=f"{get_settings().groq_model} via Groq",
        uptime_seconds=round(time.time() - _metrics.start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    total = _metrics.total_queries
    avg_latency = round(_metrics.total_latency_ms / total, 1) if total > 0 else 0.0
    return MetricsResponse(
        uptime_seconds=round(time.time() - _metrics.start_time, 1),
        total_queries=total,
        cache_hits=_metrics.cache_hits,
        confidence_distribution=_metrics.confidence_counts,
        avg_latency_ms=avg_latency,
        cache_stats=_cache.stats,
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit(get_settings().rate_limit)
def query(
    request: Request,
    req: QueryRequest,
    pipeline: ClinicalRAGPipeline = Depends(get_pipeline),
    _auth: None = Depends(verify_api_key),
) -> QueryResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    cache_key = f"{req.question}|{req.top_k}"

    # Cache hit — skip LLM entirely
    cached_result = _cache.get(cache_key)
    if cached_result is not None:
        _metrics.cache_hits += 1
        logger.info("Cache hit", extra={"request_id": request_id})
        return _build_response(cached_result, latency_ms=0.0, cached=True, request_id=request_id)

    t0 = time.perf_counter()
    try:
        result = pipeline.answer(req.question, top_k=req.top_k)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline error", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    _cache.set(cache_key, result)
    _metrics.total_queries += 1
    _metrics.confidence_counts[result.confidence] = (
        _metrics.confidence_counts.get(result.confidence, 0) + 1
    )
    _metrics.total_latency_ms += latency_ms

    logger.info(
        "Query served",
        extra={
            "request_id": request_id,
            "latency_ms": latency_ms,
            "confidence": result.confidence,
        },
    )
    return _build_response(result, latency_ms=latency_ms, cached=False, request_id=request_id)


def _build_response(
    result: RAGResponse,
    *,
    latency_ms: float,
    cached: bool,
    request_id: str,
) -> QueryResponse:
    return QueryResponse(
        question=result.query,
        answer=result.answer,
        confidence=result.confidence,
        confidence_note=result.confidence_note,
        max_score=result.max_score,
        mean_score=result.mean_score,
        sources=[SourceDoc(**s) for s in result.sources],
        latency_ms=latency_ms,
        cached=cached,
        request_id=request_id,
    )


@app.post(
    "/query/stream",
    summary="Stream a clinical RAG answer via Server-Sent Events",
    response_description="SSE stream: meta → token… → done",
)
@limiter.limit(get_settings().rate_limit)
def query_stream(
    request: Request,
    req: QueryRequest,
    pipeline: ClinicalRAGPipeline = Depends(get_pipeline),
    _auth: None = Depends(verify_api_key),
) -> StreamingResponse:
    """
    Server-Sent Events stream.  Each event is a JSON object on a `data:` line:

      data: {"type": "meta", "confidence": "HIGH", "sources": [...], ...}
      data: {"type": "token", "content": "Based on"}
      data: {"type": "token", "content": " the literature..."}
      data: {"type": "done"}
    """

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def _generate() -> AsyncGenerator:
        try:
            for event in pipeline.answer_stream(req.question, top_k=req.top_k):
                yield _sse(event)
        except EnvironmentError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
        except Exception as exc:
            logger.exception("Streaming pipeline error")
            yield _sse({"type": "error", "detail": f"Pipeline error: {exc}"})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
        },
    )
