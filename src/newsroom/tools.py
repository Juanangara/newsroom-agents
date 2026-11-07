"""Tools the Researcher agent uses: news search (Google News RSS) and article fetching."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import quote_plus

from .schemas import Article

RSS = "https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={gl}:{lang}"


def _clean(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def search_news(query: str, n: int = 10, lang: str = "es", country: str = "CO") -> list[Article]:
    import feedparser

    url = RSS.format(q=quote_plus(query), hl=f"{lang}-419", gl=country, lang=lang)
    feed = feedparser.parse(url)
    arts = []
    for e in feed.entries[: n * 2]:
        arts.append(
            Article(
                title=_clean(e.get("title", "")),
                url=e.get("link", ""),
                source=_clean(getattr(e.get("source", None), "title", "") or e.get("source", {}).get("title", "")),
                published=e.get("published", ""),
                text=_clean(e.get("summary", "")),
            )
        )
    return dedupe(arts)[:n]


def dedupe(arts: list[Article], threshold: float = 0.85) -> list[Article]:
    kept: list[Article] = []
    for a in arts:
        if not any(SequenceMatcher(None, a.title.lower(), k.title.lower()).ratio() > threshold for k in kept):
            kept.append(a)
    return kept


def fetch_text(article: Article, max_chars: int = 4000, timeout: int = 10) -> Article:
    """Try to get the full article body; fall back silently to the RSS snippet."""
    try:
        import requests
        import trafilatura

        url = article.url
        if "news.google.com" in url:  # RSS links are JS redirects: resolve to the publisher URL
            from googlenewsdecoder import gnewsdecoder

            url = gnewsdecoder(url, interval=1).get("decoded_url") or url
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        body = trafilatura.extract(r.content)  # bytes: lets trafilatura detect the charset (r.text guessed latin-1) or ""
        if len(body) > len(article.text):
            return article.model_copy(update={"text": body[:max_chars], "url": url})
    except Exception:
        pass
    return article
