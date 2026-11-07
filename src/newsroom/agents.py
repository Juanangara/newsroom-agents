"""The four agents. Each is a small class with one job and a strict JSON contract."""
from __future__ import annotations

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


class Verifier:
    """Fact-checks each key fact against the source text; unsupported facts are dropped."""

    system = (
        "You are a strict fact-checker. For each numbered claim, decide if it is directly supported "
        "by the source text. Reply with ONLY a JSON list of booleans, one per claim, in order."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, s: ArticleSummary, a: Article) -> VerifiedSummary:
        if not s.key_facts:
            return VerifiedSummary(**s.model_dump())
        claims = "\n".join(f"{i}. {f}" for i, f in enumerate(s.key_facts))
        verdicts = chat_json(self.llm, self.system, f"Source text:\n{a.text}\n\nClaims:\n{claims}", max_tokens=100)
        verdicts = (list(verdicts) + [False] * len(s.key_facts))[: len(s.key_facts)]
        ok = [f for f, v in zip(s.key_facts, verdicts) if v is True]
        bad = [f for f, v in zip(s.key_facts, verdicts) if v is not True]
        return VerifiedSummary(**s.model_dump(), supported_facts=ok, dropped_facts=bad)


class Editor:
    system = (
        "You are a news editor writing a briefing in Spanish. Use ONLY the verified facts provided. "
        "Cite every statement with its source number like [1]. Structure: a title, a 2-sentence "
        "executive summary, then bullet points grouped by theme. Do not invent anything."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, topic: str, items: list[VerifiedSummary]) -> str:
        blocks = "\n\n".join(
            f"[{i + 1}] {v.title}\nSummary: {v.summary}\nVerified facts: " + "; ".join(v.supported_facts)
            for i, v in enumerate(items)
        )
        body = self.llm.chat(self.system, f"Topic: {topic}\n\n{blocks}", max_tokens=900)
        refs = "\n".join(f"[{i + 1}] {v.title} — {v.url}" for i, v in enumerate(items))
        return f"{body}\n\n## Fuentes\n{refs}\n"
