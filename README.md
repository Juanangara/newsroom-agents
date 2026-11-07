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

## Example output
Topic "reforma pensional Colombia", Qwen2.5-7B, 16-article benchmark run (excerpt; full report in
[`results/example_report.md`](results/example_report.md)). Evidence quotes were attached offline from the
Verifier's stored records, matching what `show_quotes=True` renders.

```markdown
# Reforma pensional en Colombia: avances y controversias

La reforma pensional en Colombia fue aprobada por el Congreso sin cambios, pero enfrenta críticas sobre su impacto fiscal y la libertad de elección del ciudadano. La Corte Constitucional ordenó repetir el trámite debido a vicios de procedimiento, lo que ha generado nuevas discusiones y propuestas de contrarreforma. [1] [2] [3] [4] [5] [6] [7] [8]

- El senador Andrés Forero advirtió que la reforma pensional agravará el pasivo fiscal y comprometerá el ahorro nacional [1]
  > «El senador Andrés Forero advierte que el proyecto agravará el pasivo fiscal y comprometerá el ahorro nacional.»
- La oposición impulsa una contrarreforma enfocada en garantizar la sostenibilidad financiera del sistema y la libertad de elección del ciudadano [1]
  > «La oposición impulsa una contrarreforma enfocada en garantizar la sostenibilidad financiera del sistema y la libertad de elección del ciudadano.»
- El Ministerio de Hacienda y Colpensiones solicitaron un plazo extra para la entrada en vigencia de la reforma pensional [1]
  > «Han planteado que existe un impacto fiscal y por eso piden que se permita un plazo más largo para su entrada en vigencia.»
- El Congreso de Colombia aprobó los 9 artículos de la reforma pensional del expresidente Gustavo Petro sin modificaciones [2]
  > «La Cámara de Representantes tramita sin modificaciones los 9 artículos devueltos por la Corte Constitucional y evita una conciliación con el Senado sobre el tiempo»
- La aprobación se logró sin el apoyo del ministro del Interior del actual gobierno [2]
  > «En el debate se notó la ausencia del ministro del Interior de De la Espriella. El exministro Lara fue despedido hace una semana, y el nuevo minis»
...
```

The quotes make weak bullets visible. The fourth bullet says the approval happened "without the Interior minister's
support", while the quote only says the minister was absent: an interpretation, the kind of error that survives
the quote check (see the DEVLOG).

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
