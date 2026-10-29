"""Convert the released `sentences/*.jsonl` into MLX-LM chat data (data/mlx/{train,valid,test}.jsonl)."""
import argparse
import json
from pathlib import Path

from prompt import INSTRUCTIONS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="shabdabodha-dataset", help="local copy of the Hugging Face dataset")
ap.add_argument("--out", default="data/mlx")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for split, name in (("train", "train"), ("validation", "valid"), ("test", "test")):
    rows = [json.loads(line) for line in open(Path(args.data) / "sentences" / f"{split}.jsonl", encoding="utf-8") if line.strip()]
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            answer = json.dumps({"sb_words": r["sb_words"], "sb_sentence": r["sb_sentence"]}, ensure_ascii=False, indent=2)
            msgs = [{"role": "user", "content": INSTRUCTIONS + r["sentence"]}, {"role": "assistant", "content": answer}]
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} examples -> {out / (name + '.jsonl')}")
