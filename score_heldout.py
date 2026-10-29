"""Automatic measures for the systems in `heldout/predictions.jsonl` (20 held-out sentences)."""
import argparse
import json
from pathlib import Path

from sb_metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="shabdabodha-dataset")
args = ap.parse_args()

rows = [json.loads(line) for line in open(Path(args.data) / "heldout" / "predictions.jsonl", encoding="utf-8") if line.strip()]
gts = [r["reference_sb_sentence"] for r in rows]
print(f"{len(rows)} sentences\n{'system':20s} balanced exact outer head  F1")
for system in [c for c in rows[0] if c not in ("id", "sentence", "reference_sb_sentence")]:
    m = score([r[system] for r in rows], gts)
    print(f"{system:20s} {m['balanced'] * 100:6.1f}% {m['exact'] * 100:5.1f}% {m['outer'] * 100:5.1f}% {m['head'] * 100:5.1f}% {m['f1']:.3f}")
