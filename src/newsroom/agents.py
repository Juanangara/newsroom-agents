"""The four agents. Each is a small class with one job and a strict JSON contract."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .llm import LLM, chat_json
from .schemas import Article, ArticleSummary, VerifiedSummary
from .tools import fetch_text, search_news


class Researcher:
    """Searches the news, fetches texts and keeps only articles relevant to the topic."""

    system = (
        "You are a news researcher. Given a topic and numbered headlines, return ONLY a JSON list "
        "with the ids of headlines clearly relevant to the topic. Example: [0, 2, 5]"
    )

    def __init__(self, llm: LLM, n_candidates: int = 10, max_articles: int = 5):
        self.llm, self.n, self.max = llm, n_candidates, max_articles

    def run(self, topic: str) -> list[Article]:
        cands = search_news(topic, n=self.n)
        listing = "\n".join(f"{i}. {a.title} ({a.source})" for i, a in enumerate(cands))
        ids = chat_json(self.llm, self.system, f"Topic: {topic}\n\nHeadlines:\n{listing}", max_tokens=100)
        chosen = [cands[i] for i in ids if isinstance(i, int) and 0 <= i < len(cands)][: self.max]
        return [fetch_text(a) for a in (chosen or cands[: self.max])]


_DANGLING = {
    # Spanish function words a complete fact does not end with
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con", "por", "para",
    "que", "como", "y", "e", "o", "u", "a", "su", "sus", "lo", "se", "sobre", "entre", "desde", "hasta",
    "según", "sin", "tras", "ante", "ni", "si", "cuando", "donde", "porque",
    # English
    "the", "of", "and", "to", "in", "for", "with", "by", "from", "an", "that", "which", "or",
}
_INTERROGATIVES = {"cómo", "qué", "cuándo", "dónde", "cuál", "cuáles", "quién", "quiénes"}


def is_incomplete_fact(fact: str) -> bool:
    """Heuristic for facts the writer model cut off mid-sentence ("... explorar cómo la tecnología").

    Catches: fewer than 4 words, ending in a function word, or an interrogative opening a clause in the
    last three words. It cannot catch every truncation."""
    words = re.findall(r"[\wÁÉÍÓÚÑáéíóúñ]+", fact)
    if len(words) < 4 or words[-1].lower() in _DANGLING:
        return True
    return any(w.lower() in _INTERROGATIVES for w in words[-3:])


class Summarizer:
    system = (
        "You summarise a news article. Use ONLY information in the article. Write the summary and every "
        "fact in the SAME LANGUAGE as the article (Spanish article -> Spanish text); never translate. "
        "Reply with ONLY JSON: "
        '{"summary": "2-3 sentences", "key_facts": ["short, self-contained, verifiable fact", ...]} '
        "with at most 4 key facts, each containing concrete details (who/what/when/numbers). "
        "Every fact must be a COMPLETE sentence with a subject and a verb; never cut a sentence short."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, idx: int, a: Article) -> ArticleSummary:
        data = chat_json(self.llm, self.system, f"Title: {a.title}\n\nArticle:\n{a.text}", max_tokens=500)
        facts = [str(f).strip() for f in data.get("key_facts", [])]
        return ArticleSummary(
            article_id=idx, title=a.title, url=a.url, summary=str(data.get("summary", "")),
            key_facts=[f for f in facts if not is_incomplete_fact(f)],
            discarded_incomplete=[f for f in facts if is_incomplete_fact(f)],
        )


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%$.,]", " ", t.lower())).strip()


def quote_in_text(quote: str, text: str, min_ratio: float = 0.8) -> bool:
    """True if `quote` appears (almost) verbatim in `text`: longest common block covers >= min_ratio."""
    q, t = _norm(quote), _norm(text)
    if len(q) < 12:  # too short to be meaningful evidence
        return False
    if q in t:
        return True
    m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size / len(q) >= min_ratio


_NUM = re.compile(r"\d[\d.,]*")


def _digits(s: str) -> str:
    """Drop thousands/decimal separators so '765,000' == '765.000' == '765000'."""
    return re.sub(r"[.,]", "", s)


def numbers_missing(claim: str, text: str) -> list[str]:
    """Numbers in the claim that do not appear in `text` (a quote or the whole source).
    Separator-insensitive, and whole-number only (so '23' does not match inside '2023')."""
    t = _digits(_norm(text))
    missing = []
    for n in _NUM.findall(claim):
        d = _digits(_norm(n).strip(".,"))
        if d and not re.search(rf"(?<!\d){re.escape(d)}(?!\d)", t):
            missing.append(n)
    return missing


_EN_WORDS = re.compile(r"\b(the|of|and|to|was|were|that|in|for|with|by|from|has|have|will)\b", re.I)


def looks_english(text: str) -> bool:
    """Cheap heuristic used only for diagnostics: is this fact written in English?"""
    return len(_EN_WORDS.findall(text)) >= 2


def names_missing(claim: str, text: str) -> list[str]:
    """Capitalised words in the claim (proper-noun candidates, not the first word) absent from `text`."""
    t = _norm(text)
    words = re.findall(r"[\wÁÉÍÓÚÑáéíóúñ]+", claim)
    return [w for i, w in enumerate(words) if i > 0 and w[0].isupper() and not w.isdigit() and _norm(w) not in t]


def details_in_quote(claim: str, quote: str) -> bool:
    """Numbers and proper nouns in the claim must appear in the evidence quote.
    A real quote that does not contain the claim's specifics does not support it."""
    return not numbers_missing(claim, quote) and not names_missing(claim, quote)


def is_headline_echo(claim: str, title: str, threshold: float = 0.75) -> bool:
    """Facts that merely restate the headline carry no concrete information."""
    return SequenceMatcher(None, _norm(claim), _norm(title)).ratio() >= threshold


class EntailmentJudge:
    """Meaning-level check: does this quote, on its own, fully support this claim?

    Lexical checks cannot catch a quote on the right topic that does not support the claim (e.g. "every five
    years" vs "every six months", or a senator's warning stated as fact). This is a separate, short call."""

    system = (
        "You are a strict fact-checker. Decide whether the QUOTE, on its own, fully supports the CLAIM. "
        "Answer false if the claim adds anything the quote does not say (a number, a name, a time frame, a "
        "cause, who said it), if the claim states as fact something the quote presents as one person's "
        "opinion or warning, if the quote is about the same topic but says something different, or if the "
        "quote is cut off. The claim and the quote may be in different languages: compare meaning. "
        'Reply with ONLY JSON: {"supported": true or false, "reason": "max 12 words"}'
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, claim: str, quote: str) -> tuple[bool, str]:
        try:
            data = chat_json(self.llm, self.system, f"CLAIM: {claim}\n\nQUOTE: {quote}", max_tokens=120)
        except ValueError:
            return False, "unparseable judge reply"  # fail closed: do not keep a fact we could not check
        if not isinstance(data, dict):
            return False, "unparseable judge reply"
        return data.get("supported") is True, str(data.get("reason", ""))[:120]


class Verifier:
    """Fact-checks each key fact. The model must give a quote from the source; code checks the quote
    really exists in the text, so a sloppy 'true' from a small model is not enough."""

    system = (
        "You are a strict fact-checker. For each numbered claim, copy the EXACT sentence or fragment from "
        "the source text that supports it. If the source does not support the claim, use null. "
        'Reply with ONLY a JSON list like [{"id": 0, "quote": "exact text from source"}, {"id": 1, "quote": null}].'
    )

    def __init__(self, llm: LLM, min_quote_chars: int = 100, entailment: bool = False):
        """min_quote_chars: shorter quotes are rejected (hand labels: quotes <120 chars were supported only
        25% of the time, and truncated quotes pass every lexical check).
        entailment: also ask the model whether the quote really supports the whole fact (see EntailmentJudge)."""
        self.llm = llm
        self.min_quote_chars = min_quote_chars
        self.judge = EntailmentJudge(llm) if entailment else None

    def run(self, s: ArticleSummary, a: Article) -> VerifiedSummary:
        if not s.key_facts:
            return VerifiedSummary(**s.model_dump())
        claims = "\n".join(f"{i}. {f}" for i, f in enumerate(s.key_facts))
        rows = chat_json(self.llm, self.system, f"Source text:\n{a.text}\n\nClaims:\n{claims}", max_tokens=600)
        quotes: dict[int, str] = {}
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and isinstance(r.get("id"), int) and isinstance(r.get("quote"), str):
                quotes[r["id"]] = r["quote"]
        ok, reasons, evidence = [], {}, []
        for i, f in enumerate(s.key_facts):
            q = quotes.get(i)
            # Every signal is recorded separately so alternative filter rules can be evaluated offline.
            ev = {"article": a.title[:70], "fact": f, "quote": q, "headline_echo": is_headline_echo(f, a.title),
                  "fact_looks_english": looks_english(f),
                  "quote_in_source": None, "numbers_missing_in_quote": None, "names_missing_in_quote": None,
                  "numbers_missing_in_source": numbers_missing(f, a.text),
                  "names_missing_in_source": names_missing(f, a.text)}
            if q is None:
                reasons[f] = "no_quote"
            else:
                ev["quote_in_source"] = quote_in_text(q, a.text)
                ev["numbers_missing_in_quote"] = numbers_missing(f, q)
                ev["names_missing_in_quote"] = names_missing(f, q)
                # Proper-noun matching is recorded in the evidence but no longer rejects facts: on the
                # hand-labelled sample it raised precision 0.60 -> 0.67 while cutting recall 0.97 -> 0.40.
                if not ev["quote_in_source"]:
                    reasons[f] = "quote_not_in_source"
                elif len(q) < self.min_quote_chars:
                    reasons[f] = "quote_too_short"
                elif ev["numbers_missing_in_quote"]:
                    reasons[f] = "number_missing"
                elif ev["headline_echo"]:
                    reasons[f] = "headline_echo"
                elif self.judge is not None:
                    ev["entailed"], ev["entail_reason"] = self.judge.run(f, q)
                    if not ev["entailed"]:
                        reasons[f] = "not_entailed"
            if f not in reasons:
                ok.append(f)
            ev["supported"] = f not in reasons
            ev["reason"] = reasons.get(f)
            evidence.append(ev)
        bad = [f for f in s.key_facts if f in reasons]
        return VerifiedSummary(**s.model_dump(), supported_facts=ok, dropped_facts=bad,
                               drop_reasons=reasons, evidence=evidence)


def _inner_names(text: str) -> list[str]:
    """Capitalised words that do not start a sentence (proper-noun candidates)."""
    return re.findall(r"(?<=[^\s.!?¿¡])\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)", text)


class Editor:
    """Builds the report mostly in code, so the model cannot alter what the Verifier approved.

    - Bullets are the verified facts themselves, each with its `[n]` citation (written by code).
    - The model only proposes a title and a short summary. Both are discarded if they introduce a number
      or a proper noun that is not in the verified facts (an earlier run had it expand "Ministerio TIC" into
      a ministry name that appears nowhere in the sources).
    - The summary cites every source used, because it is derived from all of them.
    """

    system = (
        "Eres un editor de noticias. Con los hechos verificados que se te dan, responde SOLO con un JSON: "
        '{"title": "título corto", "summary": "resumen de máximo 2 frases"}. '
        "Usa exclusivamente información de los hechos: no añadas cifras, nombres propios, siglas "
        "expandidas ni explicaciones que no estén en ellos. Escribe en español."
    )

    def __init__(self, llm: LLM, show_quotes: bool = True):
        """show_quotes: print the evidence quote under each bullet so a reader can check it. About 1 in 5
        verified facts was still not supported by its quote in the hand-labelled sample."""
        self.llm = llm
        self.show_quotes = show_quotes
        self.summary_grounded = True  # False if the model's summary had to be discarded

    @staticmethod
    def _ungrounded(text: str, allowed: str) -> bool:
        """True if `text` contains a number or a proper noun that is not in `allowed`."""
        allowed_n = _norm(allowed)
        return bool(numbers_missing(text, allowed)) or any(_norm(w) not in allowed_n for w in _inner_names(text))

    def run(self, topic: str, items: list[VerifiedSummary]) -> str:
        usable = [(i, v) for i, v in enumerate(items) if v.supported_facts]
        title = topic[:1].upper() + topic[1:]
        if not usable:
            return f"# {title}\n\nNo se encontraron hechos verificables.\n"

        facts_text = " ".join(f for _, v in usable for f in v.supported_facts)
        listing = "\n".join(f"- {f}" for _, v in usable for f in v.supported_facts)
        try:
            data = chat_json(self.llm, self.system, f"Tema: {topic}\n\nHechos verificados:\n{listing}", max_tokens=300)
        except ValueError:
            data = {}
        data = data if isinstance(data, dict) else {}
        new_title, summary = str(data.get("title", "")).strip(), str(data.get("summary", "")).strip()

        allowed = f"{facts_text} {topic}"
        self.summary_grounded = True
        if summary and self._ungrounded(summary, allowed):
            summary, self.summary_grounded = "", False
        if new_title and not self._ungrounded(new_title, allowed):
            title = new_title

        bullets, seen = [], []
        for i, v in usable:
            quotes = {e["fact"]: e["quote"] for e in v.evidence if e.get("supported") and e.get("quote")}
            for f in v.supported_facts:
                text = f.strip().rstrip(".")
                if any(SequenceMatcher(None, _norm(text), _norm(s)).ratio() > 0.9 for s in seen):
                    continue  # near-duplicate of a fact already listed
                seen.append(text)
                line = f"- {text} [{i + 1}]"
                q = " ".join(quotes.get(f, "").split())
                if self.show_quotes and q:
                    q = q if len(q) <= 300 else q[:297].rstrip() + "..."
                    line += "\n  > «" + q + "»"
                bullets.append(line)

        cites = " ".join(f"[{i + 1}]" for i, _ in usable)
        parts = [f"# {title}"]
        if summary:
            parts.append(f"{summary} {cites}")
        parts.append("\n".join(bullets))
        refs = "\n".join(f"[{i + 1}] {v.title} — {v.url}" for i, v in usable)
        return "\n\n".join(parts) + f"\n\n## Fuentes\n{refs}\n"


def citation_coverage(markdown: str) -> float:
    """Fraction of content lines (before '## Fuentes') that carry a [n] citation.

    Headings, evidence quotes (lines starting with '>') and lines with fewer than 2 words besides citations (e.g. an empty '- [1]') are not
    content. A report with no content lines scores 0.0, not 1.0.
    """
    body = markdown.split("## Fuentes")[0]
    content = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ">")):
            continue
        text = re.sub(r"\[\d+\]", "", re.sub(r"^[-*•]\s*", "", line))
        if len(re.findall(r"\w+", text)) >= 2:
            content.append(line)
    return sum(bool(re.search(r"\[\d+\]", l)) for l in content) / len(content) if content else 0.0
