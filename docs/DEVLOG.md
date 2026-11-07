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

## Next steps (in order)
1. [x] Stricter verification in code (numbers / proper nouns in the quote, reject headline echoes).
2. [~] Reproducible benchmark: harness done; run `notebooks/benchmark.ipynb` on Colab and record the table here.
3. [ ] Example report and results table in the README.
4. [ ] Migrate orchestration to LangGraph with a retry loop when an article's grounding is low
       (only if the benchmark shows it helps).
