"""Measure the meaning-level judge against hand labels.

The labels (results/labeling/labels_filled.csv) say whether a quote really supports a fact. We run the
EntailmentJudge on exactly those (fact, quote) pairs and compare. No benchmark run is needed, just ~50
short model calls.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .agents import EntailmentJudge, numbers_missing
from .llm import LLM


def load_labels(csv_path: str | Path) -> dict[int, int]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        return {int(r["id"]): int(r["label"]) for r in csv.DictReader(f) if r["label"].strip() in ("0", "1")}


def _scores(pred: dict[int, bool], labels: dict[int, int]) -> dict:
    tp = sum(1 for i, l in labels.items() if l == 1 and pred[i])
    fp = sum(1 for i, l in labels.items() if l == 0 and pred[i])
    fn = sum(1 for i, l in labels.items() if l == 1 and not pred[i])
    acc = tp + fp
    return {"accepts": acc, "precision": round(tp / acc, 2) if acc else None,
            "recall": round(tp / (tp + fn), 2) if tp + fn else None, "false_accepts": fp}


def evaluate_judge(llm: LLM, evidence_json: str | Path, labels_csv: str | Path,
                   model_index: int = 0, min_quote_chars: int = 100) -> dict:
    """Score three rules on the hand-labelled facts: lexical (cheap rules), judge alone, and both."""
    evidence = json.loads(Path(evidence_json).read_text(encoding="utf-8"))[model_index]["evidence"]
    labels = load_labels(labels_csv)
    judge = EntailmentJudge(llm)

    verdicts, reasons = {}, {}
    for i in labels:
        e = evidence[i]
        verdicts[i], reasons[i] = judge.run(e["fact"], e["quote"] or "")

    def lexical(i):
        e = evidence[i]
        q = e["quote"] or ""
        return bool(q) and e["quote_in_source"] and len(q) >= min_quote_chars \
            and not numbers_missing(e["fact"], q) and not e["headline_echo"]

    lex = {i: lexical(i) for i in labels}
    out = {
        "n_labelled": len(labels),
        "base_rate_supported": round(sum(labels.values()) / len(labels), 2),
        "lexical (numbers + quote length)": _scores(lex, labels),
        "judge alone": _scores(verdicts, labels),
        "lexical AND judge": _scores({i: lex[i] and verdicts[i] for i in labels}, labels),
    }
    # keep every disagreement so it can be inspected by hand
    out["disagreements"] = [
        {"id": i, "label": labels[i], "judge": verdicts[i], "judge_reason": reasons[i],
         "fact": evidence[i]["fact"], "quote": evidence[i]["quote"]}
        for i in labels if int(verdicts[i]) != labels[i]
    ]
    return out


def print_report(res: dict) -> None:
    print(f"labelled facts: {res['n_labelled']}  (base rate supported: {res['base_rate_supported']})\n")
    print(f"{'rule':36s} accepts  precision  recall  false accepts")
    for k in ("lexical (numbers + quote length)", "judge alone", "lexical AND judge"):
        s = res[k]
        print(f"{k:36s} {s['accepts']:7d}  {s['precision']!s:>9}  {s['recall']!s:>6}  {s['false_accepts']:>13}")
    print(f"\n{len(res['disagreements'])} disagreements with your labels:")
    for d in res["disagreements"]:
        print(f"\n#{d['id']} label={d['label']} judge={d['judge']} ({d['judge_reason']})")
        print(f"  FACT : {d['fact']}\n  QUOTE: {(d['quote'] or '')[:200]}")
