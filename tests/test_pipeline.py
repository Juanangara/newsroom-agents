from newsroom import FakeLLM, run_newsroom
from newsroom import tools
from newsroom.llm import extract_json
from newsroom.schemas import Article


def test_extract_json_tolerates_fences():
    assert extract_json('Claro:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_dedupe_removes_near_duplicates():
    arts = [Article(title="Colombia aprueba reforma", url="1"), Article(title="Colombia aprueba la reforma", url="2"),
            Article(title="Otro tema distinto", url="3")]
    assert len(tools.dedupe(arts)) == 2


def test_full_pipeline_drops_unsupported_facts(monkeypatch):
    arts = [Article(title="A", url="u1", text="El dólar cerró en 4000 pesos.")]
    monkeypatch.setattr("newsroom.agents.search_news", lambda q, n=10: arts)
    monkeypatch.setattr("newsroom.agents.fetch_text", lambda a: a)
    llm = FakeLLM([
        "[0]",                                                             # researcher
        '{"summary": "Dólar estable.", "key_facts": ["cerró en 4000", "subió 10%"]}',  # summarizer
        '[{"id": 0, "quote": "El dólar cerró en 4000 pesos."}, {"id": 1, "quote": "subió diez por ciento"}]',  # verifier (2nd quote is fabricated)
        "# Informe\n- Dólar en 4000 [1]\n- Sin cita",                               # editor
    ])
    r = run_newsroom("dólar", llm, verbose=False)
    assert r.sources[0].supported_facts == ["cerró en 4000"]
    assert r.sources[0].dropped_facts == ["subió 10%"]
    assert r.grounding_rate == 0.5
    assert "## Fuentes" in r.markdown
    assert r.citation_coverage == 0.5


def test_quote_must_exist_in_source():
    from newsroom.agents import quote_in_text
    text = "El dólar cerró en 4000 pesos, según el Banco de la República."
    assert quote_in_text("El dólar cerró en 4000 pesos", text)
    assert not quote_in_text("El dólar subió un diez por ciento", text)
    assert not quote_in_text("null", text)
