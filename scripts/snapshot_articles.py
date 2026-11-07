"""Freeze a set of real articles to JSON so benchmark runs are comparable.

Usage: python scripts/snapshot_articles.py "topic" data/out.json [n]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from newsroom.tools import fetch_text, search_news  # noqa: E402

topic, out = sys.argv[1], Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
arts = [fetch_text(a) for a in search_news(topic, n=n + 4)]
arts = [a for a in arts if len(a.text) > 600][:n]  # keep only articles with a real body
out.write_text(json.dumps({"topic": topic, "articles": [a.model_dump() for a in arts]},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved {len(arts)} articles -> {out}")
