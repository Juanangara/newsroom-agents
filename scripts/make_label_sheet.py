"""Build a blind labelling sheet (CSV) from the per-fact evidence stored by the benchmark.

The sheet hides the filter's verdict so the reviewer is not anchored by it. Row ids are indexes into the
`evidence` list of the source JSON, so `score_rules.py` can join labels back to the checks.

Usage: python scripts/make_label_sheet.py results/benchmark_diag.json results/labeling/to_label.csv [model_index]
"""
import csv
import json
import random
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
model_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
evidence = json.loads(src.read_text(encoding="utf-8"))[model_idx]["evidence"]

rows = [{"id": i, "article": e["article"], "fact": e["fact"], "quote": e["quote"] or "(the verifier gave no quote)"}
        for i, e in enumerate(evidence)]
random.Random(0).shuffle(rows)  # fixed seed: same order every time, not grouped by article

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows accents correctly
    w = csv.DictWriter(f, fieldnames=["id", "article", "fact", "quote", "label", "notes"])
    w.writeheader()
    for r in rows:
        w.writerow({**r, "label": "", "notes": ""})
print(f"{len(rows)} rows -> {out}")
