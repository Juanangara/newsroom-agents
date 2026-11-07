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
    r = run_newsroom("dólar", llm, verbose=False, min_quote_chars=0)
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


def test_claim_details_must_be_in_quote():
    from newsroom.agents import details_in_quote
    quote = "El 47% de las empresas colombianas identifica la escasez de talento"
    assert details_in_quote("El 47% de las empresas identifica escasez de talento", quote)
    assert not details_in_quote("El 57% de las empresas identifica escasez de talento", quote)
    assert not details_in_quote("Según Microsoft, la escasez de talento es el problema", quote)


def test_headline_echo_is_rejected():
    from newsroom.agents import is_headline_echo
    title = "Colombia busca en la inteligencia artificial una respuesta a los retos de su sistema de salud"
    assert is_headline_echo("Colombia busca en la IA una respuesta a los retos de su sistema de salud", title)
    assert not is_headline_echo("El 47% de las empresas no encuentra personal capacitado", title)


def test_benchmark_with_fixed_articles(tmp_path):
    import json
    from newsroom.benchmark import evaluate, to_markdown_table
    snap = tmp_path / "s.json"
    snap.write_text(json.dumps({"topic": "dólar", "articles": [
        {"title": "A", "url": "u", "text": "El dólar cerró en 4000 pesos hoy."}]}), encoding="utf-8")
    llm = FakeLLM([
        '{"summary": "x", "key_facts": ["El dólar cerró en 4000 pesos"]}',
        '[{"id": 0, "quote": "El dólar cerró en 4000 pesos hoy."}]',
        "# T\n- Dólar en 4000 [1]",
    ])
    res = evaluate("fake", llm, [snap], min_quote_chars=0)
    assert res["grounding_rate"] == 1.0 and res["citation_coverage"] == 1.0
    assert res["json_failures"] == "0/1"
    assert "| fake |" in to_markdown_table([res])


def test_extract_json_single_element_list_stays_a_list():
    assert extract_json('[{"id": 0, "quote": "x"}]') == [{"id": 0, "quote": "x"}]
    assert extract_json('Sure: {"a": [1, 2]}') == {"a": [1, 2]}


def test_citation_coverage_ignores_empty_bullets_and_empty_reports():
    from newsroom.agents import citation_coverage
    assert citation_coverage("# T\nUn párrafo sin cita.\n- [1]\n- [2]\n## Fuentes\n[1] x") == 0.0
    assert citation_coverage("# T\n- Hecho concreto aquí [1]\n- Otro hecho sin cita\n") == 0.5
    assert citation_coverage("# Solo titulo\n") == 0.0


def test_drop_reasons_are_recorded():
    from newsroom.agents import Verifier
    from newsroom.schemas import ArticleSummary
    art = Article(title="Titular distinto", url="u", text="El dólar cerró en 4000 pesos hoy en Bogotá.")
    facts = ["x sin cita", "y cita falsa", "El dólar cerró en 5000 pesos", "El dólar cerró en 4000 pesos"]
    s = ArticleSummary(article_id=0, title="t", url="u", summary="", key_facts=facts)
    llm = FakeLLM(['[{"id":0,"quote":null},{"id":1,"quote":"algo que no está en el texto"},'
                   '{"id":2,"quote":"El dólar cerró en 4000 pesos hoy"},{"id":3,"quote":"El dólar cerró en 4000 pesos"}]'])
    v = Verifier(llm, min_quote_chars=0).run(s, art)
    assert v.drop_reasons["x sin cita"] == "no_quote"
    assert v.drop_reasons["y cita falsa"] == "quote_not_in_source"
    assert v.drop_reasons["El dólar cerró en 5000 pesos"] == "number_missing"
    assert v.supported_facts == ["El dólar cerró en 4000 pesos"]


def test_fixed_verifier_is_used_for_fact_checking():
    writer = FakeLLM([
        '{"summary": "x", "key_facts": ["El dólar cerró en 4000 pesos"]}',   # summarizer
        "# T\n- Dólar en 4000 [1]",                                           # editor
    ])
    checker = FakeLLM(['[{"id": 0, "quote": "El dólar cerró en 4000 pesos hoy."}]'])  # verifier only
    arts = [Article(title="A", url="u", text="El dólar cerró en 4000 pesos hoy.")]
    r = run_newsroom("dólar", writer, articles=arts, verbose=False, verifier_llm=checker, min_quote_chars=0)
    assert r.grounding_rate == 1.0
    assert len(checker.calls) == 1 and len(writer.calls) == 2


def test_numbers_are_separator_insensitive_and_whole():
    from newsroom.agents import numbers_missing
    assert numbers_missing("más de 765,000 reclamaciones", "registró 765.000 reclamaciones") == []
    assert numbers_missing("2,3 salarios", "hasta 2.3 salarios") == []
    assert numbers_missing("23 casos", "ocurrió en 2023") == ["23"]   # not a substring match
    assert numbers_missing("el 81% de las empresas", "cerca del 82% de las empresas") == ["81"]


def test_summarizer_keeps_article_language_and_english_is_flagged():
    from newsroom.agents import Summarizer, looks_english
    assert "SAME LANGUAGE" in Summarizer.system
    assert looks_english("The reform was approved in the Chamber of Representatives")
    assert not looks_english("La reforma fue aprobada en la Cámara de Representantes")


def _verify(quote, fact="El dólar cerró en 4000 pesos hoy", entailment=None, replies=None):
    """Run the Verifier on a one-fact article; `entailment` is the judge's scripted reply, if enabled."""
    from newsroom.agents import Verifier
    from newsroom.schemas import ArticleSummary
    text = ("El dólar cerró en 4000 pesos hoy, según informó el Banco de la República en su reporte diario "
            "del mercado cambiario de Bogotá, con una variación mínima frente al día anterior.")
    art = Article(title="Titular distinto", url="u", text=text)
    s = ArticleSummary(article_id=0, title="t", url="u", summary="", key_facts=[fact])
    llm = FakeLLM(['[{"id": 0, "quote": "%s"}]' % quote] + (replies or []))
    return Verifier(llm, entailment=entailment is not None).run(s, art), llm


def test_short_quote_is_rejected_by_default():
    v, _ = _verify("El dólar cerró en 4000 pesos hoy")                       # real but only 33 chars
    assert v.drop_reasons[v.key_facts[0]] == "quote_too_short"


def test_long_quote_is_accepted_and_name_mismatch_no_longer_rejects():
    long_q = ("El dólar cerró en 4000 pesos hoy, según informó el Banco de la República en su reporte "
              "diario del mercado cambiario")
    v, _ = _verify(long_q, fact="El Dólar cerró en 4000 pesos hoy en Colombia")  # 'Dólar','Colombia' not in quote
    assert v.supported_facts == ["El Dólar cerró en 4000 pesos hoy en Colombia"]
    assert v.evidence[0]["names_missing_in_quote"]                              # still recorded as a signal


def test_entailment_judge_can_reject_and_fails_closed():
    long_q = ("El dólar cerró en 4000 pesos hoy, según informó el Banco de la República en su reporte "
              "diario del mercado cambiario")
    v, llm = _verify(long_q, entailment=True, replies=['{"supported": false, "reason": "adds a time frame"}'])
    assert v.drop_reasons[v.key_facts[0]] == "not_entailed" and len(llm.calls) == 2
    v, _ = _verify(long_q, entailment=True, replies=['{"supported": true, "reason": "ok"}'])
    assert v.supported_facts
    v, _ = _verify(long_q, entailment=True, replies=["no sé", "tampoco", "nada"])   # unparseable x3 -> fail closed
    assert not v.supported_facts and v.drop_reasons[v.key_facts[0]] == "not_entailed"


def test_label_eval_scores_judge_against_labels(tmp_path):
    import csv, json
    from newsroom.label_eval import evaluate_judge
    q = "x" * 120
    ev = [{"fact": "fa", "quote": q, "quote_in_source": True, "headline_echo": False},
          {"fact": "fb", "quote": q, "quote_in_source": True, "headline_echo": False}]
    (tmp_path / "e.json").write_text(json.dumps([{"evidence": ev}]), encoding="utf-8")
    with (tmp_path / "l.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "article", "fact", "quote", "label", "notes"])
        w.writerow([0, "a", "fa", q, "1", ""]); w.writerow([1, "a", "fb", q, "0", ""])
    llm = FakeLLM(['{"supported": true, "reason": "r"}', '{"supported": false, "reason": "r"}'])
    res = evaluate_judge(llm, tmp_path / "e.json", tmp_path / "l.csv")
    assert res["judge alone"]["precision"] == 1.0 and res["judge alone"]["recall"] == 1.0
    assert res["lexical (numbers + quote length)"]["false_accepts"] == 1   # lexical alone keeps the bad one
    assert res["disagreements"] == []
