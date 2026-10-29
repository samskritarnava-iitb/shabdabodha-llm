"""Human-evaluation statistics and automatic measures from the released `human_eval/` files."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from sb_metrics import score

SYSTEMS = ["reference", "gemma_sft", "gemma_sft_ift", "llama_sft", "llama_sft_ift"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="shabdabodha-dataset")
args = ap.parse_args()
root = Path(args.data) / "human_eval"

outputs = [json.loads(line) for line in open(root / "outputs.jsonl", encoding="utf-8") if line.strip()]
ids = {r["entry_id"]: i for i, r in enumerate(outputs)}
annots = sorted({r["annotator"] for r in csv.DictReader(open(root / "scores.csv", encoding="utf-8"))})
A = np.full((len(annots), len(SYSTEMS), 2, len(outputs)), np.nan)  # annotator, system, WM/SM, entry
for r in csv.DictReader(open(root / "scores.csv", encoding="utf-8")):
    for lv, col in enumerate(("wm_score", "sm_score")):
        if r[col] != "":
            A[annots.index(r["annotator"]), SYSTEMS.index(r["system"]), lv, ids[r["entry_id"]]] = float(r[col])


def kalpha(M):
    vals, Do, n = [], 0.0, 0
    for u in range(M.shape[1]):
        v = M[:, u][~np.isnan(M[:, u])]
        if len(v) < 2:
            continue
        Do += ((v[:, None] - v[None, :]) ** 2).sum() / (len(v) - 1)
        n += len(v)
        vals.extend(v)
    vals = np.array(vals)
    return 1 - (Do / n) / (((vals[:, None] - vals[None, :]) ** 2).sum() / (n - 1) / n)


def means(idx, label):
    print(f"--- {label}: WM / SM / Overall ---")
    for si, s in enumerate(SYSTEMS):
        w, m = np.nanmean(A[idx, si, 0]), np.nanmean(A[idx, si, 1])
        print(f"  {s:14s} {w:.2f} / {m:.2f} / {(w + m) / 2:.2f}")


means(list(range(len(annots))), "pooled")
for a, name in enumerate(annots):
    means([a], name)

print("\n--- Krippendorff alpha (interval), model outputs ---")
M = A[:, 1:].reshape(len(annots), -1)
print("all:", round(kalpha(M), 3))
for a, b in itertools.combinations(range(len(annots)), 2):
    print(f"  {annots[a]}-{annots[b]}: {kalpha(M[[a, b]]):.3f}")


def wilcoxon(idx, label):
    O = (np.nanmean(A[idx][:, :, 0], axis=0) + np.nanmean(A[idx][:, :, 1], axis=0)) / 2
    pairs = list(itertools.combinations(range(1, 5), 2))
    raw = [stats.wilcoxon(O[i], O[j]).pvalue for i, j in pairs]
    holm, run = np.zeros(len(raw)), 0.0
    for rank, k in enumerate(np.argsort(raw)):
        run = max(run, min(1.0, raw[k] * (len(raw) - rank)))
        holm[k] = run
    print(f"\n--- paired Wilcoxon on per-entry overall ({label}), Holm over 6 pairs ---")
    for k, (i, j) in enumerate(pairs):
        print(f"  {SYSTEMS[i]} vs {SYSTEMS[j]}: diff {np.mean(O[i] - O[j]):+.2f}, Holm p={holm[k]:.4f}")


wilcoxon(list(range(len(annots))), "all annotators")
wilcoxon([0, 2], f"{annots[0]} + {annots[2]} only")

print("\n--- automatic measures on the 198 entries ---")
print(f"{'system':14s} balanced exact outer head  F1")
gts = [r["reference_sb_sentence"] or "" for r in outputs]
for s in SYSTEMS[1:]:
    m = score([r[f"{s}_sb_sentence"] or "" for r in outputs], gts)
    print(f"{s:14s} {m['balanced'] * 100:6.1f}% {m['exact'] * 100:5.1f}% {m['outer'] * 100:5.1f}% {m['head'] * 100:5.1f}% {m['f1']:.3f}")
