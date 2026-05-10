"""
src/rag_pipeline.py
-------------------
Full RAG pipeline:
  query → retrieve → build prompt → LLM (with retry) → answer + citations + confidence

LLM backend: Groq Llama-3-70B (free tier, <1s latency)
Set env var:  GROQ_API_KEY=your_key
Get free key: https://console.groq.com
"""

import os
from dataclasses import dataclass
from typing import List

from groq import APIConnectionError, Groq, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logger import get_logger
from src.retriever import ClinicalRetriever, RetrievalResult

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a clinical AI assistant that answers medical questions
using only the provided PubMed literature excerpts.

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say so explicitly.
- Always cite which source (by number) supports each claim.
- Keep answers concise, structured, and clinically precise.
- Never make diagnostic or treatment recommendations — state findings only.
- End every answer with a one-sentence disclaimer."""


def build_prompt(query: str, result: RetrievalResult) -> str:
    context_blocks = [
        f"[Source {i} | PubMed ID: {chunk.pubid} | Score: {chunk.score}]\n{chunk.chunk}"
        for i, chunk in enumerate(result.chunks, 1)
    ]
    return (
        f"Clinical Question: {query}\n\n"
        f"Retrieved Literature:\n{chr(10).join(context_blocks)}\n\n"
        f"Retrieval Confidence: {result.confidence} "
        f"(max similarity: {result.max_score})\n\n"
        "Answer (cite sources by number):"
    )


@dataclass
class RAGResponse:
    query: str
    answer: str
    confidence: str
    confidence_note: str
    max_score: float
    mean_score: float
    sources: List[dict]


class ClinicalRAGPipeline:
    def __init__(self, top_k: int | None = None) -> None:
        self.retriever = ClinicalRetriever(top_k=top_k)
        self._client: Groq | None = None

    def _get_client(self) -> Groq:
        if self._client is None:
            settings = get_settings()
            api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
                )
            self._client = Groq(api_key=api_key)
        return self._client

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        settings = get_settings()
        client = self._get_client()
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        return response.choices[0].message.content.strip()

    def answer_stream(self, query: str, top_k: int | None = None):
        """Stream the LLM response token-by-token as a generator.

        Yields dicts:
          {"type": "meta",  ...confidence, sources, scores}   — first, before tokens
          {"type": "token", "content": "<text>"}              — one per LLM chunk
          {"type": "done"}                                     — signals end of stream
        """
        result = self.retriever.retrieve(query, top_k=top_k)
        prompt = build_prompt(query, result)

        sources = [
            {
                "index": i + 1,
                "pubid": c.pubid,
                "label": c.label,
                "score": c.score,
                "excerpt": c.chunk[:200] + "..." if len(c.chunk) > 200 else c.chunk,
            }
            for i, c in enumerate(result.chunks)
        ]

        # Send retrieval metadata before the first token so the UI can render
        # confidence and sources immediately while tokens stream in.
        yield {
            "type": "meta",
            "confidence": result.confidence,
            "confidence_note": result.confidence_note,
            "max_score": result.max_score,
            "mean_score": result.mean_score,
            "sources": sources,
        }

        settings = get_settings()
        client = self._get_client()
        stream = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield {"type": "token", "content": delta}

        yield {"type": "done"}

    def answer(self, query: str, top_k: int | None = None) -> RAGResponse:
        """Run the full RAG pipeline.

        top_k is forwarded to the retriever per-request so concurrent callers
        with different k values never race on shared state.
        """
        logger.info("RAG query received", extra={"query": query, "top_k": top_k})

        result = self.retriever.retrieve(query, top_k=top_k)
        prompt = build_prompt(query, result)

        try:
            answer_text = self._call_llm(prompt)
        except Exception:
            logger.exception("LLM call failed")
            raise

        sources = [
            {
                "index": i + 1,
                "pubid": c.pubid,
                "label": c.label,
                "score": c.score,
                "excerpt": c.chunk[:200] + "..." if len(c.chunk) > 200 else c.chunk,
            }
            for i, c in enumerate(result.chunks)
        ]

        logger.info(
            "RAG response generated",
            extra={"confidence": result.confidence, "sources": len(sources)},
        )

        return RAGResponse(
            query=query,
            answer=answer_text,
            confidence=result.confidence,
            confidence_note=result.confidence_note,
            max_score=result.max_score,
            mean_score=result.mean_score,
            sources=sources,
        )
