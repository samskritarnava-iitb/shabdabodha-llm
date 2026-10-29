"""Automatic measures for the systems in `heldout/predictions.csv` (20 held-out sentences)."""
import argparse
import csv
from pathlib import Path

from sb_metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="shabdabodha-dataset")
args = ap.parse_args()

rows = list(csv.DictReader(open(Path(args.data) / "heldout" / "predictions.csv", encoding="utf-8")))
gts = [r["reference_sb_sentence"] for r in rows]
print(f"{len(rows)} sentences\n{'system':20s} balanced exact outer head  F1")
for system in [c for c in rows[0] if c not in ("id", "sentence", "reference_sb_sentence")]:
    m = score([r[system] for r in rows], gts)
    print(f"{system:20s} {m['balanced'] * 100:6.1f}% {m['exact'] * 100:5.1f}% {m['outer'] * 100:5.1f}% {m['head'] * 100:5.1f}% {m['f1']:.3f}")
