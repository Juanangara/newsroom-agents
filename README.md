# newsroom-agents

[![tests](https://github.com/Juanangara/newsroom-agents/actions/workflows/tests.yml/badge.svg)](https://github.com/Juanangara/newsroom-agents/actions/workflows/tests.yml)

A team of autonomous agents that produces a news report with **verified claims and cited sources**.
It runs on a free T4 GPU (Kaggle or Colab) with local open models and no API keys.

**In short:** every bullet in the report is a claim the Verifier found, verbatim, in the source article, shown
with its evidence quote. The project is also a measurement exercise: a frozen benchmark, hand-labelled data and a
[development log](docs/DEVLOG.md) of what worked and what did not (including a judge model that turned out not
to help).

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
- **Swappable LLM:** local open models in 4-bit (Qwen2.5 3B/7B, Phi-3.5-mini); other backends plug into a small `LLM` interface.
- **Robust:** JSON parsing with retries and self-correction; one failing article does not stop the pipeline.
- **Testable without a GPU:** a scripted `FakeLLM` gives deterministic tests.

## Results
Benchmark on a frozen set of 16 real Colombian news articles (2 topics), Kaggle T4, 4-bit models.
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

## Run
On Colab open `notebooks/colab.ipynb`, enable a T4 GPU and run all cells (on Kaggle use `notebooks/benchmark_kaggle.ipynb`). Or manually:

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

## Limitations
- One benchmark set (16 Colombian news articles, 2 topics) and a single annotator; small differences are noise.
- The Verifier checks that a quote exists and contains the claim's numbers; it does not check meaning. About 22%
  of verified facts were still unsupported on hand-labelled data, which is why quotes are shown.
- Spanish-language news only so far.

## Roadmap
- Retry loop (e.g. LangGraph) for low-grounding articles, only if a benchmark shows it helps.
- Larger judge model, adopted only if it clears a pre-registered threshold on the held-out labels.
- Scheduled delivery of the report by email or Telegram.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for reproducing the benchmark and the labelling setup.
