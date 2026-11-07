"""Score candidate verification rules against hand labels.

label: 1 = the quote really supports the whole fact, 0 = it does not, ? or empty = skip.
A rule "accepts" a fact when it would keep it in the report.

Precision matters most here: an accepted fact that is not supported is a false claim in the report.

Usage: python scripts/score_rules.py results/benchmark_diag.json results/labeling/to_label.csv [model_index]
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from newsroom.agents import names_missing, numbers_missing  # noqa: E402

src, sheet = Path(sys.argv[1]), Path(sys.argv[2])
model_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
evidence = json.loads(src.read_text(encoding="utf-8"))[model_idx]["evidence"]


def base(e):
    return e["quote"] is not None and e["quote_in_source"] and not e["headline_echo"]


RULES = {
    "any real quote (no detail checks)": lambda e: base(e),
    "numbers only": lambda e: base(e) and not numbers_missing(e["fact"], e["quote"]),
    "numbers + names vs source": lambda e: base(e) and not numbers_missing(e["fact"], e["quote"])
    and not e["names_missing_in_source"],
    "numbers + names vs quote (current)": lambda e: base(e) and not numbers_missing(e["fact"], e["quote"])
    and not names_missing(e["fact"], e["quote"]),
}

labels = {}
with sheet.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        lab = row["label"].strip()
        if lab in ("0", "1"):
            labels[int(row["id"])] = int(lab)

n_pos = sum(labels.values())
print(f"labelled: {len(labels)} facts ({n_pos} supported, {len(labels) - n_pos} not supported)\n")
if not labels:
    sys.exit("No labels found: fill the 'label' column with 1 or 0.")

print(f"{'rule':40s} {'accepts':>8s} {'precision':>10s} {'recall':>7s} {'false accepts':>14s}")
for name, rule in RULES.items():
    tp = sum(1 for i, l in labels.items() if l == 1 and rule(evidence[i]))
    fp = sum(1 for i, l in labels.items() if l == 0 and rule(evidence[i]))
    fn = sum(1 for i, l in labels.items() if l == 1 and not rule(evidence[i]))
    acc = tp + fp
    prec = tp / acc if acc else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"{name:40s} {acc:8d} {prec:10.2f} {rec:7.2f} {fp:14d}")
