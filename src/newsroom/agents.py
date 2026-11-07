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


class Summarizer:
    system = (
        "You summarise a news article. Use ONLY information in the article. Reply with ONLY JSON: "
        '{"summary": "2-3 sentences", "key_facts": ["short, self-contained, verifiable fact", ...]} '
        "with at most 4 key facts, each containing concrete details (who/what/when/numbers)."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, idx: int, a: Article) -> ArticleSummary:
        data = chat_json(self.llm, self.system, f"Title: {a.title}\n\nArticle:\n{a.text}", max_tokens=500)
        return ArticleSummary(
            article_id=idx, title=a.title, url=a.url,
            summary=str(data.get("summary", "")), key_facts=[str(f) for f in data.get("key_facts", [])],
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


def numbers_missing(claim: str, text: str) -> list[str]:
    """Numbers in the claim that do not appear in `text` (a quote or the whole source)."""
    t = _norm(text)
    return [n for n in _NUM.findall(claim) if _norm(n).strip(".,") not in t]


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


class Verifier:
    """Fact-checks each key fact. The model must give a quote from the source; code checks the quote
    really exists in the text, so a sloppy 'true' from a small model is not enough."""

    system = (
        "You are a strict fact-checker. For each numbered claim, copy the EXACT sentence or fragment from "
        "the source text that supports it. If the source does not support the claim, use null. "
        'Reply with ONLY a JSON list like [{"id": 0, "quote": "exact text from source"}, {"id": 1, "quote": null}].'
    )

    def __init__(self, llm: LLM):
        self.llm = llm

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
                  "quote_in_source": None, "numbers_missing_in_quote": None, "names_missing_in_quote": None,
                  "numbers_missing_in_source": numbers_missing(f, a.text),
                  "names_missing_in_source": names_missing(f, a.text)}
            if q is None:
                reasons[f] = "no_quote"
            else:
                ev["quote_in_source"] = quote_in_text(q, a.text)
                ev["numbers_missing_in_quote"] = numbers_missing(f, q)
                ev["names_missing_in_quote"] = names_missing(f, q)
                if not ev["quote_in_source"]:
                    reasons[f] = "quote_not_in_source"
                elif ev["numbers_missing_in_quote"]:
                    reasons[f] = "number_missing"
                elif ev["names_missing_in_quote"]:
                    reasons[f] = "name_missing"
                elif ev["headline_echo"]:
                    reasons[f] = "headline_echo"
            if f not in reasons:
                ok.append(f)
            ev["supported"] = f not in reasons
            ev["reason"] = reasons.get(f)
            evidence.append(ev)
        bad = [f for f in s.key_facts if f in reasons]
        return VerifiedSummary(**s.model_dump(), supported_facts=ok, dropped_facts=bad,
                               drop_reasons=reasons, evidence=evidence)


class Editor:
    system = (
        "Eres un editor de noticias. Redacta un informe breve en español usando EXCLUSIVAMENTE los hechos "
        "verificados que se te dan. Formato: un título (#), un resumen ejecutivo de 2 frases, y luego viñetas "
        "con '- '. CADA frase y CADA viñeta debe terminar con la cita de su fuente, por ejemplo [1] o [2]. "
        "No añadas nada que no esté en los hechos."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, topic: str, items: list[VerifiedSummary]) -> str:
        usable = [(i, v) for i, v in enumerate(items) if v.supported_facts]
        if not usable:
            return f"# {topic}\n\nNo se encontraron hechos verificables.\n"
        blocks = "\n\n".join(
            f"Fuente [{i + 1}] {v.title}\nHechos verificados:\n" + "\n".join(f"- {f}" for f in v.supported_facts)
            for i, v in usable
        )
        body = self.llm.chat(self.system, f"Tema: {topic}\n\n{blocks}", max_tokens=900)
        refs = "\n".join(f"[{i + 1}] {v.title} — {v.url}" for i, v in usable)
        return f"{body}\n\n## Fuentes\n{refs}\n"


def citation_coverage(markdown: str) -> float:
    """Fraction of content lines (before '## Fuentes') that carry a [n] citation.

    Headings and lines with fewer than 2 words besides citations (e.g. an empty '- [1]') are not
    content. A report with no content lines scores 0.0, not 1.0.
    """
    body = markdown.split("## Fuentes")[0]
    content = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        text = re.sub(r"\[\d+\]", "", re.sub(r"^[-*•]\s*", "", line))
        if len(re.findall(r"\w+", text)) >= 2:
            content.append(line)
    return sum(bool(re.search(r"\[\d+\]", l)) for l in content) / len(content) if content else 0.0
