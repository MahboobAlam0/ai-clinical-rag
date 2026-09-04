"""
src/train_embeddings.py
-----------------------
Fine-tune the sentence-transformer embedding model on PubMedQA
question→context pairs using MultipleNegativesRankingLoss (MNRL).

MNRL treats every other example in the batch as a negative, so a large
batch size is important — use 32 or 64 if your GPU/RAM allows.

Usage:
    python src/train_embeddings.py
    python src/train_embeddings.py --epochs 5 --batch-size 64 --max-pairs 4000

After training, rebuild the FAISS index with the new model:
    python src/ingest.py --model models/clinical-embeddings

GPU is recommended but not required — CPU training is slower (~10 min/epoch).
"""

import argparse
import json
import os
import random
from pathlib import Path

# Disable W&B before any transformers/sentence-transformers imports so the
# HuggingFace Trainer never tries to initialise a wandb run.
os.environ.setdefault("WANDB_DISABLED", "true")

from sentence_transformers import InputExample, SentenceTransformer, evaluation
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "pubmed_contexts.jsonl"
DEFAULT_OUTPUT = Path("models") / "clinical-embeddings"


# ── Data preparation ──────────────────────────────────────────────────────────

def load_training_pairs(
    path: Path,
    max_pairs: int | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[InputExample], list[InputExample]]:
    """
    Build (question, context) positive pairs from PubMedQA JSONL.

    Each document already pairs a question with its supporting abstract,
    making it ideal supervision signal for dense retrieval training.
    Only 'yes'/'no' labels are used — 'maybe' answers are ambiguous.
    """
    random.seed(seed)
    examples: list[InputExample] = []

    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("decision", "").lower() not in ("yes", "no"):
                continue
            context = doc["context"].strip()
            question = doc["question"].strip()
            if not context or not question:
                continue
            examples.append(InputExample(texts=[question, context]))

    random.shuffle(examples)
    if max_pairs:
        examples = examples[:max_pairs]

    split = max(1, int(len(examples) * (1 - val_ratio)))
    train_examples = examples[:split]
    val_examples = examples[split:]

    logger.info(
        "Training data prepared",
        extra={"train": len(train_examples), "val": len(val_examples)},
    )
    return train_examples, val_examples


def build_ir_evaluator(
    val_examples: list[InputExample],
) -> evaluation.InformationRetrievalEvaluator:
    """
    Build an Information Retrieval evaluator from the validation set.

    Reports nDCG@10 and MRR@10 after each epoch so you can track
    whether retrieval quality actually improves.
    """
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for i, ex in enumerate(val_examples):
        qid = f"q{i}"
        did = f"d{i}"
        queries[qid] = ex.texts[0]
        corpus[did] = ex.texts[1]
        relevant_docs[qid] = {did}

    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="pubmedqa-val",
        show_progress_bar=False,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    warmup_ratio: float,
    max_pairs: int | None,
) -> None:
    logger.info(
        "Starting embedding fine-tune",
        extra={
            "base_model": base_model,
            "output_dir": str(output_dir),
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    train_examples, val_examples = load_training_pairs(DATA_PATH, max_pairs=max_pairs)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,   # MNRL requires full batches
    )

    evaluator = build_ir_evaluator(val_examples)

    # ── Load model ────────────────────────────────────────────────────────────
    model = SentenceTransformer(base_model)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # MNRL: given a batch of (query, passage) pairs, every other passage in
    # the batch acts as an in-batch negative. No manual negative mining needed.
    loss = MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    logger.info("Warmup steps", extra={"warmup_steps": warmup_steps})

    # ── Fit ───────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=len(train_dataloader),   # evaluate once per epoch
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,                      # keeps the checkpoint with best nDCG
        show_progress_bar=True,
        checkpoint_path=str(output_dir / "checkpoints"),
        checkpoint_save_steps=len(train_dataloader),
    )

    logger.info("Training complete", extra={"model_saved": str(output_dir)})
    print(
        f"\n✅ Fine-tuned model saved to: {output_dir}\n\n"
        "Next step — rebuild the FAISS index with the new embeddings:\n"
        f"  python src/ingest.py --model {output_dir}\n\n"
        "Then update EMBEDDING_MODEL in your .env:\n"
        f"  EMBEDDING_MODEL={output_dir}\n"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Fine-tune the clinical embedding model on PubMedQA pairs."
    )
    parser.add_argument(
        "--base-model",
        default=settings.embedding_model,
        help="HuggingFace model ID or local path to start from (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the fine-tuned model (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size — larger = better MNRL negatives (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of steps used for linear LR warmup (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap training pairs for quick experiments (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        max_pairs=args.max_pairs,
    )
