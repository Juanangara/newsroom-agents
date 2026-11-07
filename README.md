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
| **Verifier** | Must quote the source for each fact; code checks the quote exists and contains the claim's numbers and names. Unsupported facts are **dropped** | `[{id, quote}]` |
| **Editor** | Writes the report from verified facts only, with `[n]` citations | markdown |

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

| Model | Grounding | Facts supported / proposed | JSON failures | Time (s) |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.18 | 9 / 51 | 0/16 | 524 |
| Qwen2.5-7B-Instruct | **0.32** | **21 / 65** | 0/16 | 597 |
| Phi-3.5-mini-instruct | 0.20 | 14 / 69 | 1/16 | 843 |

Caveats: single run, and each model verifies its own facts, so the numbers mix generation and
verification quality. The full analysis, bugs found along the way and what is still unknown are in
[`docs/DEVLOG.md`](docs/DEVLOG.md).

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
