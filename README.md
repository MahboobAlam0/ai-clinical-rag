# Clinical RAG

> Retrieval-Augmented Generation over PubMed literature — with domain-adapted embeddings, uncertainty quantification, and a production-grade API.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama--3.3--70B-orange)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What this is

A clinical question-answering system that retrieves evidence from 5,355 PubMed abstracts and generates grounded answers using Llama-3.3-70B. Every answer includes **cited sources**, a **confidence score** derived from retrieval signal strength, and a **disclaimer** — the model is explicitly prevented from answering outside the retrieved context.

The embedding model was fine-tuned on PubMedQA using MultipleNegativesRankingLoss, improving retrieval nDCG@10 from **0.806 → 0.926** on a held-out test set.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  SentenceTransformer                │  Fine-tuned on PubMedQA
│  (clinical-embeddings)              │  all-MiniLM-L6-v2 base
└──────────────┬──────────────────────┘
               │  384-dim normalised vector
               ▼
┌─────────────────────────────────────┐
│  Qdrant Vector Store                │  5,355 PubMed chunks
│  Cosine similarity · top-k search  │  Local or Docker-hosted
└──────────────┬──────────────────────┘
               │  Retrieved chunks + scores
               ▼
┌─────────────────────────────────────┐
│  Uncertainty Estimator              │  HIGH / MEDIUM / LOW
│  max cosine score → confidence      │  based on retrieval signal
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
  LRU Cache        Groq Llama-3.3-70B
  (cache hit)      tenacity retry · stream support
       │                │
       └───────┬─────────┘
               ▼
┌─────────────────────────────────────┐
│  FastAPI Response                   │
│  answer · sources · confidence      │
│  latency · request_id · cached      │
└─────────────────────────────────────┘
```

---

## Fine-Tuning Results

The embedding model was fine-tuned for 10 epochs on 4,500 PubMedQA (question → context) positive pairs using **MultipleNegativesRankingLoss**. Best checkpoint saved at epoch 9 via `InformationRetrievalEvaluator`.

Evaluated on a **held-out test split** (n=300, seed=99 — separate from the training seed=42):

| Metric | Base model | Fine-tuned | Improvement |
|--------|:----------:|:----------:|:-----------:|
| Accuracy@1 | 0.700 | **0.823** | +12.3% |
| Accuracy@3 | 0.837 | **0.990** | +15.3% |
| Accuracy@5 | 0.873 | **0.997** | +12.3% |
| Accuracy@10 | 0.903 | **1.000** | +9.7% |
| MRR@10 | 0.774 | **0.900** | +12.6% |
| nDCG@10 | 0.806 | **0.926** | +11.98% |
| MAP@100 | 0.778 | **0.900** | +12.2% |

> Reproduce: `python -m eval.compare_models`

The gains are consistent across every metric and generalise to unseen data, confirming the model learned domain-relevant similarity rather than memorising training pairs.

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Domain-adapted retrieval** | Embedding model fine-tuned on PubMedQA; nDCG@10 +12% over baseline |
| **Uncertainty quantification** | Confidence (HIGH / MEDIUM / LOW) derived from max cosine similarity |
| **Streaming responses** | `POST /query/stream` returns Server-Sent Events — token by token |
| **Query caching** | 256-slot LRU cache; identical queries skip the LLM entirely |
| **Retry logic** | Tenacity exponential back-off on Groq `RateLimitError` / `ConnectionError` |
| **Request tracing** | Every request carries a `X-Request-ID` header for log correlation |
| **Structured logging** | JSON log lines with `ts`, `level`, `logger`, `request_id`, and custom fields |
| **Optional auth** | `X-API-Key` header enforcement; disabled when `API_KEY` env var is empty |
| **Rate limiting** | Per-IP limit via slowapi (default: 30 req/min, configurable) |
| **Metrics endpoint** | `GET /metrics` — uptime, query count, cache hit rate, confidence distribution |
| **Vector store** | Qdrant — local file persistence by default, Docker for production |
| **Thread-safe design** | `top_k` is forwarded per-request, never mutated on a shared object |

---

## Dataset

**PubMedQA** (`qiaojin/PubMedQA` on HuggingFace)
- 1,000 expert-labelled + 61,000 unlabelled QA pairs from PubMed abstracts
- 5,355 overlapping character-level chunks (400 chars, 80-char overlap)
- License: MIT — no API key required
- Paper: [Jin et al., 2019 — PubMedQA: A Dataset for Biomedical Research QA](https://arxiv.org/abs/1909.06146)

---

## Uncertainty Quantification

Confidence is derived from the **maximum cosine similarity** across all retrieved chunks — a lightweight, calibration-free signal that degrades gracefully when the query falls outside the corpus distribution.

| Level | Max cosine score | Meaning |
|-------|:----------------:|---------|
| 🟢 HIGH | ≥ 0.70 | Strong evidence in retrieved literature |
| 🟡 MEDIUM | ≥ 0.50 | Partial evidence — verify with a clinician |
| 🔴 LOW | < 0.50 | Weak retrieval signal — answer may be unreliable |

---

## Quick Start

```bash
git clone https://github.com/MahboobAlam0/clinical-rag
cd clinical-rag

pip install -r requirements.txt

cp .env.example .env          # add your GROQ_API_KEY

python -m src.ingest          # build Qdrant index (≈ 25 s)

uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## Full Setup

### 1. Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_key_here          # https://console.groq.com — free tier
EMBEDDING_MODEL=models/clinical-embeddings   # or omit to use base model
```

### 2. Dataset + index

```bash
# Download PubMedQA (first time only)
python data/fetch_dataset.py

# Build Qdrant vector index
python -m src.ingest
```

### 3. (Optional) Fine-tune the embedding model

```bash
python -m src.train_embeddings --epochs 10 --batch-size 32

# Rebuild index with the fine-tuned model
python -m src.ingest --model models/clinical-embeddings

# Compare base vs fine-tuned on a held-out test set
python -m eval.compare_models
```

### 4. Run

```bash
# API
uvicorn api.main:app --reload --port 8000

# Streamlit UI
python -m streamlit run app/streamlit_app.py

# Both via Docker (includes Qdrant server)
docker-compose up --build
```

---

## API Reference

### `POST /query`

Standard request — returns the full answer once the LLM finishes.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the evidence for metformin in type 2 diabetes?", "top_k": 5}'
```

```json
{
  "question": "What is the evidence for metformin in type 2 diabetes?",
  "answer": "Based on [Source 1] and [Source 3], metformin significantly reduces HbA1c...",
  "confidence": "HIGH",
  "confidence_note": "Strong evidence found in retrieved literature.",
  "max_score": 0.8412,
  "mean_score": 0.7631,
  "sources": [
    {
      "index": 1,
      "pubid": "19900953",
      "label": "RESULTS",
      "score": 0.8412,
      "excerpt": "Metformin reduced HbA1c by 1.12% compared to placebo..."
    }
  ],
  "latency_ms": 874.3,
  "cached": false,
  "request_id": "3f2c1a7e-84b1-4f9a-a3d2-001122334455"
}
```

### `POST /query/stream`

Server-Sent Events — streams tokens as they are generated, with retrieval metadata arriving before the first token.

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the evidence for metformin in type 2 diabetes?"}'
```

```
data: {"type": "meta", "confidence": "HIGH", "max_score": 0.84, "sources": [...]}

data: {"type": "token", "content": "Based"}
data: {"type": "token", "content": " on"}
data: {"type": "token", "content": " [Source 1]..."}

data: {"type": "done"}
```

### `GET /health`

```json
{ "status": "ok", "index_size": 5355, "model": "llama-3.3-70b-versatile via Groq", "uptime_seconds": 142.3 }
```

### `GET /metrics`

```json
{
  "uptime_seconds": 142.3,
  "total_queries": 47,
  "cache_hits": 12,
  "confidence_distribution": { "HIGH": 31, "MEDIUM": 14, "LOW": 2 },
  "avg_latency_ms": 912.4,
  "cache_stats": { "hits": 12, "misses": 35, "size": 35 }
}
```

**Optional API key auth** — set `API_KEY` in `.env`, then pass the header:

```bash
curl -H "X-API-Key: your-key" ...
```

---

## Project Structure

```
clinical-rag/
├── src/
│   ├── config.py            # Pydantic BaseSettings — all config from .env
│   ├── logger.py            # JSON structured logging with request ID correlation
│   ├── ingest.py            # Chunk, embed, upsert to Qdrant
│   ├── retriever.py         # Qdrant search + uncertainty estimation
│   ├── rag_pipeline.py      # Retrieve → prompt → LLM (retry + streaming)
│   └── train_embeddings.py  # Fine-tune embedding model with MNRL
├── api/
│   └── main.py              # FastAPI: /query, /query/stream, /health, /metrics
├── app/
│   └── streamlit_app.py     # Interactive UI with query history
├── eval/
│   └── compare_models.py    # Base vs fine-tuned comparison on held-out test set
├── data/
│   ├── fetch_dataset.py     # Download PubMedQA from HuggingFace
│   └── pubmed_contexts.jsonl
├── models/
│   └── clinical-embeddings/ # Fine-tuned model checkpoint (best epoch)
├── test/
│   ├── test_pipeline.py     # Unit tests: chunking, retrieval, RAG pipeline
│   └── test_api.py          # Integration tests: all endpoints via TestClient
├── .env.example
├── Dockerfile
├── docker-compose.yml       # API + Streamlit + Qdrant
└── CI.yml                   # GitHub Actions: test + docker build
```

---

## Tests

```bash
pytest test/ -v --cov=src --cov-report=term-missing
```

- **Unit tests** — chunking logic, retrieval confidence classification, RAG pipeline (LLM mocked)
- **Integration tests** — all API endpoints via `httpx` TestClient (Qdrant + LLM mocked)
- **Coverage gate** — 80% enforced in CI

---

## Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 | Fine-tuned on PubMedQA |
| Vector Store | Qdrant | Local persistence or Docker |
| LLM | Groq Llama-3.3-70B Versatile | Free tier, ~1 s, tenacity retry |
| Config | Pydantic Settings | All values from `.env` |
| API | FastAPI + Pydantic | `/query`, `/query/stream`, `/health`, `/metrics` |
| Auth | `X-API-Key` header | Disabled when `API_KEY` is empty |
| Rate Limiting | slowapi | 30 req/min per IP (configurable) |
| Caching | In-process LRU (256 slots) | Keyed by question + top_k |
| Logging | JSON structured | Request ID correlated across logs |
| UI | Streamlit | Query history, adjustable top-k |
| Containers | Docker + Compose | API, Streamlit, Qdrant as services |
| CI/CD | GitHub Actions | Tests on push, Docker build verification |

---

## Disclaimer

> This system is intended for **research and educational purposes only**. It is not a substitute for professional clinical judgment, diagnosis, or treatment advice. Always consult a qualified healthcare provider for medical decisions.
