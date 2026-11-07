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

Measured results and the issues found along the way are in [`docs/DEVLOG.md`](docs/DEVLOG.md).

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
