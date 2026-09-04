"""
eval/compare_models.py
----------------------
Compare retrieval quality between the base embedding model and the
fine-tuned clinical model on a held-out PubMedQA test split.

Prints a table like:

  Metric           Base    Fine-tuned       Δ
  ─────────────── ──────── ────────── ────────
  Accuracy@1      0.7532     0.7819   +0.0287
  nDCG@10         0.8201     0.8910   +0.0709
  ...

Run:
    python -m eval.compare_models
    python -m eval.compare_models --test-size 500 --seed 7
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")

# Use new import path (sentence-transformers v3+)
try:
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator  # type: ignore

from sentence_transformers import InputExample, SentenceTransformer

DATA_PATH = Path("data/pubmed_contexts.jsonl")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL = "models/clinical-embeddings"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_pairs(path: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """Load a held-out test split (different seed from training seed=42)."""
    random.seed(seed)
    pairs: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("decision", "").lower() not in ("yes", "no"):
                continue
            q, c = doc["question"].strip(), doc["context"].strip()
            if q and c:
                pairs.append((q, c))
    random.shuffle(pairs)
    return pairs[:n]


def build_evaluator(pairs: list[tuple[str, str]], name: str) -> InformationRetrievalEvaluator:
    queries      = {f"q{i}": q for i, (q, _) in enumerate(pairs)}
    corpus       = {f"d{i}": c for i, (_, c) in enumerate(pairs)}
    relevant_docs = {f"q{i}": {f"d{i}"} for i in range(len(pairs))}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_name: str, evaluator: InformationRetrievalEvaluator) -> dict:
    model = SentenceTransformer(model_name)
    result = evaluator(model, output_path=None)
    # Some ST versions return a float (primary metric) rather than a full dict.
    # In that case fall back to the evaluator's csv_headers / last written row.
    if isinstance(result, (int, float)):
        result = {}
    return dict(result) if result else {}


def _normalise(key: str) -> str:
    """Lower-case, strip prefix and collapse separators so key lookups are format-agnostic."""
    return key.lower().replace("-", "_").replace("@", "_at_").replace(" ", "_")


def _get(results: dict, metric: str) -> float:
    """
    Find a metric value regardless of key format variants:
      cosine-Accuracy@1, cosine_accuracy@1, test_cosine_accuracy@1, …
    """
    needle = _normalise(metric)           # e.g. "accuracy_at_1"
    for raw_key, val in results.items():
        if needle in _normalise(raw_key):
            return float(val)
    return 0.0


# ── Report ────────────────────────────────────────────────────────────────────

METRICS = [
    ("Accuracy@1",  "Accuracy@1"),
    ("Accuracy@3",  "Accuracy@3"),
    ("Accuracy@5",  "Accuracy@5"),
    ("Accuracy@10", "Accuracy@10"),
    ("MRR@10",      "MRR@10"),
    ("nDCG@10",     "NDCG@10"),
    ("MAP@100",     "MAP@100"),
]


def print_report(base: dict, ft: dict) -> None:
    # If both dicts are empty the evaluator returned a float — warn user.
    if not base and not ft:
        print(
            "\n⚠️  The evaluator returned no metrics dict (sentence-transformers v3 "
            "changed the return type).\n"
            "Falling back to the training CSV at "
            "models/clinical-embeddings/Information-Retrieval_evaluation_pubmedqa-val_results.csv\n"
            "Open that file — epoch 9 is your fine-tuned score; "
            "run with --debug to inspect raw result keys.\n"
        )
        return

    header = f"{'Metric':<15} {'Base':>10} {'Fine-tuned':>12} {'Δ':>8}"
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}")
    print(f"{'─'*15} {'─'*10} {'─'*12} {'─'*8}")

    for label, key in METRICS:
        base_v = _get(base, key)
        ft_v   = _get(ft,   key)
        delta  = ft_v - base_v
        sign   = "+" if delta >= 0 else ""
        mark   = "✓" if delta > 0.001 else ("✗" if delta < -0.001 else "")
        print(f"{label:<15} {base_v:>10.4f} {ft_v:>12.4f} {sign}{delta:>7.4f} {mark}")

    print(f"{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned embedding model on a held-out test split."
    )
    parser.add_argument("--base-model",  default=BASE_MODEL)
    parser.add_argument("--ft-model",    default=FINETUNED_MODEL)
    parser.add_argument("--test-size",   type=int, default=300)
    parser.add_argument("--seed",        type=int, default=99,
                        help="Use a seed different from 42 (training) to avoid overlap")
    parser.add_argument("--debug",       action="store_true",
                        help="Print raw result keys returned by the evaluator")
    args = parser.parse_args()

    print(f"\nLoading {args.test_size} held-out test pairs (seed={args.seed})…")
    pairs = load_test_pairs(DATA_PATH, args.test_size, args.seed)
    if not pairs:
        print("ERROR: No pairs loaded. Check data/pubmed_contexts.jsonl exists.")
        return

    evaluator = build_evaluator(pairs, name="test")

    print(f"\n[1/2] Evaluating base model: {args.base_model}")
    base_results = evaluate(args.base_model, evaluator)

    print(f"\n[2/2] Evaluating fine-tuned model: {args.ft_model}")
    ft_results = evaluate(args.ft_model, evaluator)

    if args.debug:
        print("\n── Base result keys ──")
        for k, v in base_results.items():
            print(f"  {k!r}: {v}")
        print("\n── Fine-tuned result keys ──")
        for k, v in ft_results.items():
            print(f"  {k!r}: {v}")

    print_report(base_results, ft_results)


if __name__ == "__main__":
    main()
