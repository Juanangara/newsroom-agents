# Hand-labelling sheet

`to_label.csv`: 51 facts proposed by Qwen2.5-3B (verified by Qwen2.5-7B) with the evidence quote the
verifier gave. The filter's verdict is hidden on purpose.

Fill the `label` column:

| label | meaning |
|---|---|
| `1` | The quote supports the **whole** fact: every specific (who / what / number / date) is in the quote or follows directly from it |
| `0` | The quote does not support the fact: unrelated, only partly supporting, or the fact adds something the quote does not say |
| `?` or empty | Unsure: the row is skipped |

Notes:
- Judge the **meaning**, not the language. Many facts are in English and the quotes in Spanish.
- A quote that supports part of a fact (e.g. the number but not the claim around it) is a `0`.
- Use `notes` for anything odd. Be strict: a `1` means you would be happy to see it in the report.

Then score the candidate rules:

```bash
python scripts/score_rules.py results/benchmark_diag.json results/labeling/to_label.csv
```

Precision is the metric that matters: an accepted fact that is not supported is a false claim in the report.
