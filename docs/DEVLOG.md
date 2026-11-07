# Development log

A record of each iteration, the results measured on Colab (T4, Qwen2.5-3B-Instruct in 4-bit) and the
decisions taken. Test topic: "inteligencia artificial en Colombia" (the pipeline produces Spanish reports).

## Iteration 1: first end-to-end pipeline
- Orchestration in plain Python + pydantic (no LangChain/LlamaIndex: the flow is linear and every agent
  has an explicit JSON contract).
- Environment issues solved while moving to Colab:
  - `pip install -e` was not visible to the running kernel -> the notebook uses `sys.path` instead.
  - `apply_chat_template(return_tensors="pt")` returns a `BatchEncoding` in newer `transformers`
    versions -> use `tokenize=False` + `tokenizer(...)`.
  - Google News RSS links are JS redirects -> resolved with `googlenewsdecoder`
    (text per article: ~100 -> ~4000 characters).
- **Result: grounding rate 0.24.**

### Findings
1. **Design flaw:** the Editor received the unverified `summary`, so unverified claims reached the report
   and bypassed the Verifier's filter.
2. The 3B model ignored the instruction to cite `[n]` in the body and left English headings.
3. A 3B verifier that answers a bare `true/false` is unreliable; it was impossible to tell a Summarizer
   hallucination from an over-strict verifier.

## Iteration 2: evidence-based verification
- The Verifier must **quote the source verbatim**; code checks that the quote exists (fuzzy match >= 0.8).
  A lazy `true` is no longer enough.
- The Editor receives **only verified facts**, with Spanish instructions for the output.
- New metric: `citation_coverage` (fraction of bullets carrying a `[n]` citation).
- **Result: grounding rate 0.61** (up from 0.24); every bullet cited.

### Findings
1. **A real quote does not imply it supports the claim.** One nonsensical bullet ("the service uses
   security to block suspicious actions...") passed the filter.
2. **Weak facts:** some are just the headline reworded ("Colombia seeks an answer to its health-system
   challenges in AI") and carry no concrete information.
3. **Formatting:** the 3B model omits the `#` title and the executive summary, and mixes citations into
   the first paragraph.
4. **Not reproducible:** Google News returns different articles between runs, so models cannot be compared.

## Iteration 3: stricter verification (in code)
- `details_in_quote`: numbers and proper nouns in a claim must appear in its evidence quote.
- `is_headline_echo`: claims that merely restate the article title are dropped.
- Unit tests for both. (Not yet re-measured on Colab.)

## Iteration 4: benchmark harness
- `scripts/snapshot_articles.py` freezes real articles to JSON (`data/`: 2 topics x 8 articles) so every
  model sees identical inputs; `run_newsroom(..., articles=...)` skips the Researcher.
- `newsroom.benchmark` runs each model over the snapshots and reports grounding, facts supported/proposed,
  citation coverage, JSON failures and time; results are saved after every model (Colab disconnects).
- Bug found while writing its test: `extract_json` returned the inner object for a one-element list
  (`[{"id":0,...}]`), so the Verifier silently lost its answer. Fixed + regression test.
- Models compared (no paid APIs): Qwen2.5-3B, Qwen2.5-7B, Phi-3.5-mini, all 4-bit. **Not yet run.**

## Benchmark run 1 (Colab T4, 4-bit, greedy decoding, 2 topics x 8 articles = 16 articles)

| Model | Grounding | Facts supported / proposed | JSON failures | Time (s) |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.18 | 9 / 51 | 0/16 | 524 |
| Qwen2.5-7B-Instruct | **0.32** | **21 / 65** | 0/16 | 597 |
| Phi-3.5-mini-instruct | 0.20 | 14 / 69 | 1/16 | 843 |

Raw output: `results/benchmark_run1.json` and `results/benchmark_run1.md`.

### Findings
1. **Model size helped.** The 7B model supported more than twice as many facts as the 3B (21 vs 9) with a
   higher rate, and followed the output format. Phi-3.5-mini proposed the most facts but supported few and
   was the slowest.
2. **The `citation_coverage` column of this run is invalid (it showed 1.00 for every model).** The old
   metric returned 1.0 for a report with no bullets and counted empty bullets such as `- [1]` as cited.
   Qwen 3B actually produced exactly that on one topic (a paragraph plus empty `- [1]`, `- [2]` bullets) and
   listed a source ([7]) it never used. Removed from the table above; the metric was fixed (iteration 5).
3. **Format-following differs a lot between models**, independently of fact quality: Phi-3.5 appended a stray
   "Citas:" block and an "Las fuentes: [3], [4]" line; Qwen 3B ignored the bullet format.
4. **Verification has limits.** A garbled claim from the 7B report ("an increase of 100% in financial
   obligations over the last decades") passed: a real quote that contains the numbers can still be
   misread. The checks are necessary but not sufficient.

### Caveats (read before quoting these numbers)
- Each model verifies **its own** facts, so "grounding" mixes generation quality with verification quality.
  A 3B model may be a worse verifier, not only a worse summarizer.
- Single run, greedy decoding, 16 articles: differences of a few points are not significant.
- We do not yet know *why* ~70-80% of facts are dropped (hallucination vs. a strict filter). Run 1 did not
  record drop reasons.

## Iteration 5: measurement fixes (after run 1)
- `citation_coverage` now counts only content lines (headings and empty bullets are ignored) and scores a
  report with no content as 0.0.
- Every dropped fact now records a reason (`no_quote`, `quote_not_in_source`, `details_missing`,
  `headline_echo`); the benchmark aggregates them in `drop_reasons`.
- 10 unit tests, including regressions for both. **Not yet re-run on Colab.**

## Iteration 6: fixed verifier for the benchmark
- `run_newsroom(..., verifier_llm=...)` lets a different model do the fact-checking. The benchmark can
  now run in two configurations: **fixed verifier** (Qwen2.5-7B checks every model's facts, isolating
  generation quality) and **self-verification** (default pipeline). Results go to
  `results/benchmark_<tag>.json|md`, with a `verifier` column and per-model `drop_reasons`.
- Test confirms the verifier model receives only the fact-checking call. **Not yet run on Colab.**

## Benchmark run 2: fixed verifier (Qwen2.5-7B fact-checks every model)

| Model | Grounding | Facts supported / proposed | Citation coverage | JSON failures | Time (s) |
|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.31 | 16 / 51 | 0.88 | 0/16 | 543 |
| Qwen2.5-7B-Instruct | 0.32 | 21 / 65 | 1.00 | 0/16 | 595 |
| Phi-3.5-mini-instruct | **0.41** | **28 / 69** | 0.94 | 1/16 | 878 |

Drop reasons (same run): `details_missing` 32 / 36 / 37, `quote_not_in_source` 1 / 5 / 3, `no_quote` 2 / 3 / 1
(3B / 7B / Phi).

### Findings
1. **Run 1's conclusion ("model size helped") was confounded by verifier quality.** With the same 7B verifier
   the 3B goes 0.18 -> 0.31 and Phi 0.20 -> 0.41; the ranking changes. Self-verification numbers should not be
   used to compare generators.
2. **Sanity check passed:** the 7B scores identically (21 / 65) as its own verifier in both runs, and the
   facts proposed per model (51 / 65 / 69) are identical across runs (greedy decoding is deterministic).
3. **~90% of dropped facts were dropped by our own number / proper-noun filter (`details_missing`)**, not by
   missing or invented quotes (1-5 per model). So the 0.3-0.4 grounding mostly measures how strict that
   filter is; it is **not** evidence that 60-70% of facts are hallucinated. The filter is probably too strict
   (e.g. a claim says "Colombia" while the quote says "el pais"), but this is unverified: the run stored
   only counts, not the facts.
4. Open question: Phi may lead because it writes more extractive facts that are easier to back with a quote,
   not because it is a better model.

## Iteration 7: store evidence per fact
- Each fact now records its quote and every check separately (`quote_in_source`, numbers / names missing
  in the quote and in the whole source, headline echo). `details_missing` is split into `number_missing`
  and `name_missing`. The benchmark saves all of it under `evidence`, so alternative filter rules can be
  evaluated offline without GPU time.
- New "diagnostic run" cell in the benchmark notebook (3B only, ~15 min).

## Next steps (in order)
1. [x] Stricter verification in code (numbers / proper nouns in the quote, reject headline echoes).
2. [x] Reproducible benchmark: harness + run 1 done (see above).
3. [ ] Re-run the benchmark with the fixed metric and drop reasons; use one fixed verifier (e.g. the 7B)
       for all models to separate generation quality from verification quality.
4. [ ] Example report in the README.
5. [ ] Migrate orchestration to LangGraph with a retry loop when an article's grounding is low
       (only if the benchmark shows it helps).
