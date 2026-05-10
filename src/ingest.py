"""
src/ingest.py
-------------
Loads pubmed_contexts.jsonl, chunks documents, embeds them with
sentence-transformers, and upserts into a Qdrant collection.

Run:
    python -m src.ingest
    python -m src.ingest --model models/clinical-embeddings
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import get_settings
from src.logger import get_logger
from src.retriever import _build_qdrant_client

logger = get_logger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "pubmed_contexts.jsonl"


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]


def load_documents(path: Path, size: int, overlap: int) -> Tuple[List[str], List[dict]]:
    chunks, metadata = [], []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            for chunk in chunk_text(doc["context"], size, overlap):
                chunks.append(chunk)
                metadata.append({
                    "pubid":    doc["pubid"],
                    "question": doc["question"],
                    "label":    doc["label"],
                    "decision": doc["decision"],
                    "chunk":    chunk,
                })
    logger.info("Documents loaded", extra={"chunks": len(chunks)})
    return chunks, metadata


def embed(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    logger.info("Embedding chunks", extra={"count": len(chunks)})
    return model.encode(
        chunks,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def upsert_to_qdrant(
    client: QdrantClient,
    collection: str,
    embeddings: np.ndarray,
    metadata: List[dict],
    batch_size: int = 256,
) -> None:
    dim = embeddings.shape[1]

    # Drop and recreate the collection so re-runs are idempotent
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        client.delete_collection(collection)
        logger.info("Dropped existing collection", extra={"collection": collection})

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Collection created", extra={"collection": collection, "dim": dim})

    # Upsert in batches
    total = len(metadata)
    for start in tqdm(range(0, total, batch_size), desc="Upserting to Qdrant"):
        end = min(start + batch_size, total)
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload=metadata[i],
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=points)

    logger.info("Upsert complete", extra={"vectors": total, "collection": collection})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant index from PubMedQA JSONL.")
    parser.add_argument(
        "--model",
        default=None,
        help="Override embedding model (HuggingFace ID or local path).",
    )
    args = parser.parse_args()

    settings = get_settings()
    model_name = args.model or settings.embedding_model

    logger.info("Starting ingestion", extra={"data_path": str(DATA_PATH), "model": model_name})

    chunks, metadata = load_documents(DATA_PATH, settings.chunk_size, settings.chunk_overlap)
    model = SentenceTransformer(model_name)
    embeddings = embed(chunks, model)

    client = _build_qdrant_client(settings)
    upsert_to_qdrant(client, settings.qdrant_collection, embeddings, metadata)

    print(
        f"\n✅ Indexed {len(chunks)} vectors into Qdrant "
        f"collection '{settings.qdrant_collection}'\n"
    )


if __name__ == "__main__":
    main()
