# Labelling rubric

One question: **if I only had this quote, would I confidently assert this whole fact?**
Judge whether the quote supports the fact, not whether the fact is true in the world.

## `1` - supported (all must hold)
- Every element of the fact (who, what, number, date, place, cause) is in the quote or follows directly from it.
- Paraphrases and synonyms are fine. Language is irrelevant (English fact, Spanish quote).
- The fact may be **vaguer** than the quote, never more specific ("entered in 2026" vs "early 2026" is `1`).

## `0` - not supported (any one is enough). Put the tag in `notes`.
| Tag | When |
|---|---|
| `add_info` | The fact adds something the quote does not say (a cause, consequence, or a loaded adjective). |
| `number_date` | A number, date or time frame is missing or different ("every five years" vs "every six months"). |
| `attribution` | Someone *says or warns* and the fact states it as true. |
| `actor` | The subject is changed or confused (reform vs counter-reform, government vs opposition). |
| `hedge` | A hedge is lost: "proposes" -> "approved", "could" -> "will", "seeks" -> "achieves". |
| `generalization` | "Some" -> "all", or one case -> a rule. |
| `topic_only` | Same topic, but the quote does not say it. |
| `truncated` | The quote is cut off or too fragmentary. |
| `compound` | Multi-part fact where only one part is supported. |

## `?` - skipped
- The subject is only identified through a pronoun or an antecedent outside the quote.
- Cannot decide in ~30 seconds, or the quote is unreadable (corrupt text).

## Procedure
1. Split the fact into pieces (who / what / number / date).
2. Look for each piece in the quote. If one is missing: `0`, with its tag in `notes` (only on `0`s).
3. Do not open the article or search the web.
