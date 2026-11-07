# newsroom-agents

A team of autonomous agents that produces a news report with **verified claims and cited sources**.
It runs entirely on Google Colab's free T4 GPU, with no API keys required.

```
topic ─► Researcher ─► Summarizer ─► Verifier ─► Editor ─► report.md
         (RSS + filter)   (JSON)     (fact-check)  ([n] citations)
```

| Agent | Job | Contract |
|---|---|---|
| **Researcher** | Searches Google News (RSS), deduplicates, filters by relevance and fetches the full text | list of ids (JSON) |
| **Summarizer** | Summarizes each article and extracts verifiable facts | `{summary, key_facts[]}` |
| **Verifier** | Must quote the source for each fact; code checks the quote exists in the source, is at least 100 characters and contains the claim's numbers. Unsupported facts are **dropped**. An optional LLM judge (`entailment=True`, off by default) did **not** improve precision on held-out labels with a 7B model - see the DEVLOG | `[{id, quote}]` |
| **Editor** | Bullets are the verified facts themselves, each with its `[n]` citation, assembled in code. Each bullet shows its evidence quote underneath. The model only proposes a title and short summary, discarded if they add numbers or names not in the facts | markdown |

## Why it is more than a demo
- **Explicit verification:** the report only contains facts the Verifier found in the source.
- **Grounding metric:** `report.grounding_rate` = supported facts / proposed facts.
- **Citation coverage:** `report.citation_coverage` = bullets with a `[n]` citation / all bullets.
- **Full trace** of every step (`report.trace`).
- **Swappable LLM:** local Qwen2.5 in 4-bit (Colab) or Claude via API.
- **Robust:** JSON parsing with retries and self-correction; one failing article does not stop the pipeline.
- **Testable without a GPU:** a scripted `FakeLLM` gives deterministic tests.

## Results
Benchmark on a frozen set of 16 real Colombian news articles (2 topics), Colab T4, 4-bit models.
**Grounding** = facts the Verifier could back with a verbatim quote from the source / facts proposed.

Fixed verifier (Qwen2.5-7B fact-checks for every model), current pipeline, deterministic runs:

| Model | Grounding | Facts supported / proposed | Citation coverage | JSON failures |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.83 | 40 / 48 | 1.00 | 0/16 |
| Qwen2.5-7B-Instruct | 0.71 | 46 / 65 | 1.00 | 0/16 |
| Phi-3.5-mini-instruct | 0.80 | 56 / 70 | 1.00 | 0/16 |

> Earlier numbers (0.18-0.41) were dominated by bugs later fixed (English facts, number separators, strict
> name filter) and are kept only as history in the DEVLOG.

What this shows, and what it does not:
- Grounding is the verifier's pass rate, **not** accuracy: on hand-labelled data about 22% of verified facts
  were still unsupported, which is why each bullet shows its evidence quote.
- Runs are deterministic: repeating a configuration gives identical numbers.
- Citation coverage is 1.00 by construction (bullets are built in code), so it no longer separates models.
- With 48-70 facts per model, the differences between models are within noise; no winner is claimed.
- Single set of 16 articles; small differences are not significant.

Full analysis, bugs found along the way and open questions: [`docs/DEVLOG.md`](docs/DEVLOG.md).

## Run on Colab
Open `notebooks/colab.ipynb`, enable a T4 GPU and run all cells. Or manually:

```python
!git clone https://github.com/Juanangara/newsroom-agents.git
%cd newsroom-agents
!pip install -q pydantic feedparser requests trafilatura googlenewsdecoder accelerate bitsandbytes transformers

import sys; sys.path.insert(0, "src")
from newsroom import HFLocalLLM, run_newsroom
report = run_newsroom("inteligencia artificial en Colombia", HFLocalLLM())
print(report.grounding_rate, report.citation_coverage)
print(report.markdown)
```

## Development
```bash
pip install -e '.[dev]' && pytest
```

## Roadmap
- Reproducible benchmark: Qwen 3B vs 7B vs Claude on a fixed article set.
- Retry loop (LangGraph) when an article's grounding is low.
- Scheduled delivery of the report by email or Telegram.
