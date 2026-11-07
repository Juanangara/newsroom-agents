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

## Benchmark run 3: self-verification again, with the fixed metric

| Model | Grounding | Facts supported / proposed | Citation coverage | JSON failures | Time (s) |
|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.18 | 9 / 51 | 0.75 | 0/16 | 524 |
| Qwen2.5-7B-Instruct | 0.32 | 21 / 65 | 1.00 | 0/16 | 584 |
| Phi-3.5-mini-instruct | 0.20 | 14 / 69 | 1.00 | 1/16 | 830 |

- Grounding and fact counts are **identical to run 1**: results reproduce exactly.
- Citation coverage is now meaningful (the run-1 value of 1.00 for all models was an artefact). Only
  Qwen 3B drops below 1.0. Coverage is not comparable across configurations, because the Editor sees
  different verified facts depending on who verifies.
- The metric does not penalise Phi's stray "Citas:" lines, since they contain `[n]`; it measures citation
  presence, not format cleanliness.

## Iteration 7: store evidence per fact
- Each fact now records its quote and every check separately (`quote_in_source`, numbers / names missing
  in the quote and in the whole source, headline echo). `details_missing` is split into `number_missing`
  and `name_missing`. The benchmark saves all of it under `evidence`, so alternative filter rules can be
  evaluated offline without GPU time.
- New "diagnostic run" cell in the benchmark notebook (3B only, ~15 min).

## Diagnostic run: what the dropped facts actually were (Qwen 3B writer, Qwen 7B verifier, 51 facts)
Raw data: `results/benchmark_diag.json` (`evidence` holds the quote and every check for each fact).

Drop reasons: `name_missing` 26, `number_missing` 6, `no_quote` 2, `quote_not_in_source` 1; 16 supported.

### Findings
1. **32 of the 51 facts were written in English although every article is Spanish.** The Summarizer prompt
   is English and the 3B model answered in English. A proper-noun check of an English claim ("Constitutional
   Court", "Congress", "April") against a Spanish quote can never pass. This was a pipeline bug, not
   hallucination.
2. **Number formatting caused false drops:** `765,000` in the fact vs `765.000` in the quote (also `40,316`
   vs `40.316`).
3. **The name filter is not simply "too strict".** Offline re-evaluation of alternative rules on the same 51
   facts:

   | Rule | Supported |
   |---|---|
   | current (numbers + names vs quote) | 16 / 51 |
   | + separator-insensitive numbers | 18 / 51 |
   | names checked against the whole source instead of the quote | 25 / 51 |
   | numbers only, no name check | 45 / 51 |

   Dropping the name check entirely admits unsupported claims, e.g. "the reform avoided a conflict with
   the Senate despite requests from the Ministry of Finance", backed by a quote about fiscal cost that says
   nothing of the sort. Some of what the name check caught was genuine.
4. **Consequence:** the grounding numbers in runs 1-3 mix model quality with these two bugs. The fixed-verifier
   ranking (Phi > 7B > 3B) should be treated as provisional until re-measured.
5. We cannot choose between filter rules without ground truth: a hand-labelled sample of facts
   (supported / not supported) would let us measure precision and recall of each rule.

## Iteration 8: fix the root causes
- The Summarizer must write in the article's language (never translate).
- Numbers are compared ignoring thousands/decimal separators and as whole numbers (`23` no longer matches
  inside `2023`).
- Evidence now also records `fact_looks_english` to catch any language mismatch that remains.
- 13 unit tests. **Not yet re-run on Colab.**

## Hand-labelled evaluation of the verification rules (49 facts, one labeller)
Labels: `results/labeling/labels_filled.csv` (30 supported, 19 not supported, 2 skipped). Facts from the
diagnostic run (Qwen 3B writer, Qwen 7B verifier); 32 of the 51 facts were still in English.
Scored with `scripts/score_rules.py`. A rule "accepts" a fact when it would keep it in the report.

| Rule | Accepts | Precision | Recall | False accepts |
|---|---|---|---|---|
| any real quote (no detail checks) | 48 | 0.60 | 0.97 | 19 |
| numbers only | 44 | 0.64 | 0.93 | 16 |
| numbers + quote >= 100 chars | 39 | 0.69 | 0.90 | 12 |
| numbers + quote >= 120 chars | 37 | 0.70 | 0.87 | 11 |
| numbers + names vs source | 24 | 0.67 | 0.53 | 8 |
| numbers + names vs quote (current) | 18 | 0.67 | 0.40 | 6 |

### Findings
1. **Base rate:** 61% (30/49) of the verifier's quotes really support their fact. The 7B verifier alone lets
   through a lot of unsupported claims.
2. **The name filter is nearly uncorrelated with truth.** It raises precision only from 0.60 to 0.67 while
   cutting recall from 0.97 to 0.40: of the 33 facts it rejected, 20 were genuinely supported (16 of them
   `name_missing`). That is a bad trade. The earlier hypothesis ("too strict") is confirmed by data.
3. **Quote length is the best cheap signal found:** quotes under 120 characters were supported 2/8 times
   (25%) against 28/41 (68%) for longer ones. Truncated quotes ("...trabajamos conjuntam") pass every lexical
   check. Small sample, but consistent.
4. **The remaining false accepts are semantic, not lexical:** a quote on the right topic that does not
   support the claim (e.g. fact "every five years" vs quote "every six months"; the number was written as a
   word, so the digit check could not see it). Only a meaning-level check (does the quote entail the fact?)
   can catch these.
5. **Misattribution:** facts that drop who said something (a senator's warning stated as fact, a
   counter-reform described as "the reform") are labelled unsupported but pass lexical checks.

### Caveats
Single labeller, 49 facts, English-contaminated facts, and the rules were tuned looking at this same
sample (risk of overfitting). Treat the numbers as directional.

## Iteration 9: act on the labelled evaluation
- The proper-noun check no longer rejects facts (still recorded as a signal in `evidence`).
- Quotes shorter than 100 characters are rejected (`quote_too_short`; `min_quote_chars` is a parameter).
- New optional `EntailmentJudge` (`entailment=True`): a separate call that asks whether the quote alone fully
  supports the claim. It fails closed (a fact is dropped if the judge's reply cannot be parsed).
- `newsroom.label_eval` + `notebooks/judge_eval.ipynb` measure the judge against the 49 hand labels without
  re-running the benchmark. 17 unit tests. **Judge not yet measured on Colab; it is off by default until it is.**
- Caveat: 100 characters was chosen by looking at the same sample it is evaluated on.

## Entailment judge vs. hand labels (Qwen2.5-7B judge, 49 labelled facts)
Raw output: `results/judge_eval.json` (includes all 17 disagreements with the labels).

| Rule | Accepts | Precision (95% CI) | Recall | False accepts |
|---|---|---|---|---|
| no filter (verifier's quote only) | 49 | 0.61 [0.47, 0.74] | 1.00 | 19 |
| lexical (numbers + quote length >= 100) | 39 | 0.69 [0.54, 0.81] | 0.90 | 12 |
| judge alone | 27 | 0.74 [0.55, 0.87] | 0.67 | 7 |
| lexical AND judge | 21 | **0.81** [0.60, 0.92] | 0.57 | 4 |

### Findings
1. **The judge helps, but not decisively.** Precision rises with each layer (0.61 -> 0.69 -> 0.74 -> 0.81),
   yet the 95% intervals overlap almost completely at n = 49, so none of the steps is a statistically
   established improvement. Every extra layer costs recall (0.90 -> 0.57).
2. **The 7B judge is noisy and its stated reasons cannot be trusted.** Several false rejections cite
   differences that do not exist: "year in claim differs from quote" (both say 2025), "different number of
   articles" for 93 of 95 vs 93 of 95. In other cases it nitpicks ("early 2026" vs "2026").
3. **It misses what the lexical rules catch, and vice versa.** The judge accepted a fact with the number
   "2,3 salarios minimos" absent from the quote (the number check catches it) and a senator's warning stated
   as fact (nothing lexical can). Hence "lexical AND judge" has the best precision.
4. **Part of the disagreement is labeller strictness, not judge error.** Some rows labelled unsupported are
   arguable (e.g. "takes effect on April 1, 2027" vs "vigencia diferida al 1 de abril de 2027"). One labeller
   and no inter-annotator agreement: label noise alone can move these numbers by several points.
5. Facts were still mostly English against Spanish quotes, which probably hurts a small judge.

### Decision
Keep `entailment=False` as the default. `lexical AND judge` is available (`entailment=True`) as a
high-precision / low-recall mode, appropriate when a false claim is worse than a missing one.
Not adopting the judge by default avoids claiming an improvement the data do not establish.

## Next steps (in order)
1. [x] Stricter verification in code (numbers / proper nouns in the quote, reject headline echoes).
2. [x] Reproducible benchmark: harness + run 1 done (see above).
3. [ ] Re-run the benchmark with the fixed metric and drop reasons; use one fixed verifier (e.g. the 7B)
       for all models to separate generation quality from verification quality.
4. [ ] Example report in the README.
5. [ ] Migrate orchestration to LangGraph with a retry loop when an article's grounding is low
       (only if the benchmark shows it helps).
