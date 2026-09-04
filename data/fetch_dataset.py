"""
Dataset: PubMedQA
Source : https://huggingface.co/datasets/qiaojin/PubMedQA
License: MIT
Size   : ~1,000 expert-labelled + 61,000 unlabelled PubMed abstracts
Use    : Free, no API key required

Run:
    python data/fetch_dataset.py
Output:
    data/pubmed_contexts.jsonl   -- one doc per line, ready for ingestion
"""

import json
import os
from datasets import load_dataset


def fetch_and_save(split: str = "train", max_docs: int = 5000):
    print(f"Downloading PubMedQA ({split})...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)

    out_path = os.path.join(os.path.dirname(__file__), "pubmed_contexts.jsonl")
    saved = 0

    with open(out_path, "w") as f:
        for row in ds:
            # Each row has: pubid, question, context (dict with labels/contexts/meshes), long_answer, final_decision
            contexts = row.get("context", {})
            contexts_list = contexts.get("contexts", [])
            labels = contexts.get("labels", [])

            for ctx, label in zip(contexts_list, labels):
                if not ctx.strip():
                    continue
                doc = {
                    "pubid":    row["pubid"],
                    "question": row["question"],
                    "context":  ctx.strip(),
                    "label":    label,          # METHODS / RESULTS / CONCLUSIONS etc.
                    "answer":   row.get("long_answer", ""),
                    "decision": row.get("final_decision", ""),
                }
                f.write(json.dumps(doc) + "\n")
                saved += 1
                if saved >= max_docs:
                    break
            if saved >= max_docs:
                break

    print(f"Saved {saved} documents → {out_path}")
    return out_path


if __name__ == "__main__":
    fetch_and_save()