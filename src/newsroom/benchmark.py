"""Compare local models on a fixed article set. Same inputs, same prompts, only the model changes."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from .llm import LLM
from .pipeline import run_newsroom
from .schemas import Article


def load_snapshot(path: str | Path) -> tuple[str, list[Article]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["topic"], [Article(**a) for a in data["articles"]]


def evaluate(model_name: str, llm: LLM, snapshots: list[str | Path],
             verifier_llm: LLM | None = None, verifier_name: str = "self",
             entailment: bool = False, min_quote_chars: int = 100) -> dict:
    """Run the team on every snapshot and aggregate the metrics.

    With `verifier_llm`, `llm` writes (summaries + report) but a fixed model does the fact-checking,
    which separates generation quality from verification quality."""
    proposed = supported = 0
    cov, failed, total_articles, reports = [], 0, 0, []
    reasons: dict[str, int] = {}
    evidence: list[dict] = []
    t0 = time.time()
    for path in snapshots:
        topic, arts = load_snapshot(path)
        r = run_newsroom(topic, llm, articles=arts, verbose=False, verifier_llm=verifier_llm,
                         entailment=entailment, min_quote_chars=min_quote_chars)
        n_fail = sum(1 for s in r.trace if s["agent"] == "error")
        failed += n_fail
        total_articles += len(arts)
        proposed += sum(len(v.supported_facts) + len(v.dropped_facts) for v in r.sources)
        supported += sum(len(v.supported_facts) for v in r.sources)
        for v in r.sources:
            evidence.extend({"topic": topic, **e} for e in v.evidence)
            for why in v.drop_reasons.values():
                reasons[why] = reasons.get(why, 0) + 1
        cov.append(r.citation_coverage)
        reports.append({"topic": topic, "markdown": r.markdown})
    return {
        "model": model_name,
        "verifier": verifier_name,
        "grounding_rate": round(supported / proposed, 3) if proposed else 0.0,
        "facts_proposed": proposed,
        "facts_supported": supported,
        "citation_coverage": round(sum(cov) / len(cov), 3) if cov else 0.0,
        "json_failures": f"{failed}/{total_articles}",
        "seconds": round(time.time() - t0),
        "drop_reasons": reasons,
        "evidence": evidence,
        "reports": reports,
    }


def to_markdown_table(results: list[dict]) -> str:
    head = "| Model | Verifier | Grounding | Facts supported / proposed | Citation coverage | JSON failures | Time (s) |\n"
    head += "|---|---|---|---|---|---|---|\n"
    rows = "".join(
        f"| {r['model']} | {r.get('verifier', 'self')} | {r['grounding_rate']:.2f} | "
        f"{r['facts_supported']} / {r['facts_proposed']} | "
        f"{r['citation_coverage']:.2f} | {r['json_failures']} | {r['seconds']} |\n"
        for r in results
    )
    return head + rows


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_benchmark(model_ids: list[str], snapshots: list[str | Path], out_dir: str | Path = "results",
                  verifier_id: str | None = None, tag: str = "self",
                  entailment: bool = False) -> list[dict]:
    """Benchmark each model. With `verifier_id`, that model fact-checks for everyone (loaded once, kept on
    the GPU next to the model being tested). Results go to `benchmark_<tag>.json/.md`."""
    from .llm import HFLocalLLM

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    verifier = HFLocalLLM(verifier_id) if verifier_id else None
    results = []
    for mid in model_ids:
        print(f"== {mid} (verifier: {verifier_id or 'self'})")
        llm = verifier if mid == verifier_id else HFLocalLLM(mid)
        results.append(evaluate(mid, llm, snapshots, verifier_llm=verifier,
                                verifier_name=verifier_id or "self", entailment=entailment))
        if llm is not verifier:
            del llm
        free_gpu()
        # save after every model so a Colab disconnect doesn't lose finished runs
        (out / f"benchmark_{tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / f"benchmark_{tag}.md").write_text(to_markdown_table(results), encoding="utf-8")
    print(to_markdown_table(results))
    return results
