from __future__ import annotations

import time

from .agents import Editor, citation_coverage, Researcher, Summarizer, Verifier
from .llm import LLM
from .schemas import Article, Report


def run_newsroom(topic: str, llm: LLM, max_articles: int = 5, verbose: bool = True,
                 articles: list[Article] | None = None) -> Report:
    """Run the team. Pass `articles` to skip the Researcher (fixed inputs, e.g. for benchmarks)."""
    trace: list[dict] = []

    def step(agent: str, **info):
        trace.append({"agent": agent, "t": round(time.time(), 2), **info})
        if verbose:
            print(f"[{agent}] " + ", ".join(f"{k}={v}" for k, v in info.items()))

    if articles is None:
        articles = Researcher(llm, max_articles=max_articles).run(topic)
        step("researcher", articles=len(articles), titles=[a.title[:50] for a in articles])
    else:
        step("researcher", skipped=True, articles=len(articles))

    summarizer, verifier = Summarizer(llm), Verifier(llm)
    verified = []
    for i, a in enumerate(articles):
        try:
            s = summarizer.run(i, a)
            v = verifier.run(s, a)
        except ValueError as e:  # model never produced valid JSON: skip the article, keep going
            step("error", article=i, error=str(e)[:80])
            continue
        step("summarizer+verifier", article=i, kept=len(v.supported_facts), dropped=len(v.dropped_facts))
        verified.append(v)

    md = Editor(llm).run(topic, verified)
    cov = citation_coverage(md)
    step("editor", chars=len(md), citation_coverage=round(cov, 2))

    facts = sum(len(v.supported_facts) + len(v.dropped_facts) for v in verified)
    ok = sum(len(v.supported_facts) for v in verified)
    return Report(topic=topic, markdown=md, sources=verified, grounding_rate=ok / facts if facts else 1.0,
                  citation_coverage=cov, trace=trace)
