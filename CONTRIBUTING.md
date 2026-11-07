# Contributing / reproducing

## Tests (no GPU, no network)
```bash
pip install -e '.[dev]' && pytest
```
Agents are tested with a scripted `FakeLLM`, so the suite is deterministic.

## Reproducing the benchmark
1. Open `notebooks/benchmark_kaggle.ipynb` on Kaggle (GPU T4 x2, Internet on) and run all cells (~40 min).
   Colab works too with `notebooks/benchmark.ipynb`.
2. Inputs are frozen article snapshots in `data/` (made with `scripts/snapshot_articles.py`), so runs are
   deterministic and comparable.
3. Output: `out/benchmark_final.json` (metrics, per-fact `evidence`, rendered reports) and `.md`.

## Hand labels
`results/labeling/` holds the hand-labelled facts (`labels_filled*.csv`) and the written rubric (`RUBRIC.md`).
Rules and judges are scored against them with `scripts/score_rules.py` and `src/newsroom/label_eval.py`.
Round 2 was not used to tune anything, so it serves as held-out validation.

## Layout
- `src/newsroom/`: agents, pipeline, LLM backends, benchmark harness
- `docs/DEVLOG.md`: what was tried, what the data showed, decisions
- `results/`: current results; `results/history/` keeps superseded runs (their numbers were affected by bugs
  fixed later, see the DEVLOG)

## Changing the Verifier
Evaluate rule changes offline against the stored evidence and labels before spending GPU time, and report
precision with confidence intervals. Do not tune on the held-out round.
