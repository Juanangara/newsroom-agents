from __future__ import annotations

from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str
    url: str
    source: str = ""
    published: str = ""
    text: str = ""  # full text if it could be fetched, otherwise the RSS snippet


class ArticleSummary(BaseModel):
    article_id: int
    title: str
    url: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    # facts the writer model left unfinished; removed before verification
    discarded_incomplete: list[str] = Field(default_factory=list)


class VerifiedSummary(ArticleSummary):
    supported_facts: list[str] = Field(default_factory=list)
    dropped_facts: list[str] = Field(default_factory=list)
    # fact -> why it was dropped: no_quote | quote_not_in_source | number_missing | name_missing | headline_echo
    drop_reasons: dict[str, str] = Field(default_factory=dict)
    # one record per fact with the quote and every check, to re-evaluate filter rules offline
    evidence: list[dict] = Field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        total = len(self.supported_facts) + len(self.dropped_facts)
        return 1.0 if total == 0 else len(self.supported_facts) / total


class Report(BaseModel):
    topic: str
    markdown: str
    sources: list[VerifiedSummary]
    grounding_rate: float
    citation_coverage: float = 0.0
    trace: list[dict] = Field(default_factory=list)
