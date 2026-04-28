"""Top-1 retrieval comparison: AVLMaps class-label cosine vs TextBridge
caption cosine, on a battery of *abstract* / intent-level queries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from textbridge.captioner import CAPTIONS_PATH
from textbridge.query import QUERIES, rank_sources
from textbridge.scene import SOURCES, attach_captions


def _load_captions() -> Dict[str, str]:
    if not CAPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{CAPTIONS_PATH} not found. Run `python -m textbridge.captioner` "
            "first to generate Qwen2.5-Omni-7B captions."
        )
    return json.loads(CAPTIONS_PATH.read_text())


def evaluate() -> Dict:
    captions = _load_captions()
    sources = attach_captions(captions)
    sid2src = {s.sid: s for s in sources}

    rows: List[Dict] = []
    avl_top1 = bridge_top1 = 0
    for query, gt in QUERIES:
        avl_ranked    = rank_sources(query, sources, pipeline="avlmaps")
        bridge_ranked = rank_sources(query, sources, pipeline="textbridge")
        avl_pick    = avl_ranked[0][0]
        bridge_pick = bridge_ranked[0][0]
        ok_avl    = avl_pick.sid    == gt
        ok_bridge = bridge_pick.sid == gt
        avl_top1    += int(ok_avl)
        bridge_top1 += int(ok_bridge)
        rows.append({
            "query": query, "gt": gt, "gt_class": sid2src[gt].class_label,
            "avlmaps_pick":    avl_pick.class_label,
            "textbridge_pick": bridge_pick.class_label,
            "avlmaps_caption_pick_id":    avl_pick.sid,
            "textbridge_pick_id":         bridge_pick.sid,
            "ok_avlmaps":    ok_avl,
            "ok_textbridge": ok_bridge,
            "avlmaps_cos":    avl_ranked[0][1],
            "textbridge_cos": bridge_ranked[0][1],
        })
    n = len(rows)
    return {
        "n": n,
        "avlmaps_top1":    avl_top1 / n,
        "textbridge_top1": bridge_top1 / n,
        "rows": rows,
        "captions": captions,
        "sources": sources,
    }


def format_report(result: Dict) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append(" Top-1 abstract-instruction retrieval")
    lines.append("    AVLMaps baseline: cosine(SBERT(query), SBERT(class_label))")
    lines.append("    TextBridge      : cosine(SBERT(query), SBERT(Qwen2.5-Omni-7B caption))")
    lines.append("=" * 96)
    lines.append("")
    lines.append("Sources:")
    for s in result["sources"]:
        lines.append(f"  {s.sid}  class={s.class_label:18s} room={s.room:13s}")
        lines.append(f"        Qwen-caption: {s.caption}")
    lines.append("")
    header = f"  {'query':<46} {'GT':<14} {'AVLMaps':<18} {'TextBridge':<20}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in result["rows"]:
        ma = "✓" if r["ok_avlmaps"]    else "✗"
        mt = "✓" if r["ok_textbridge"] else "✗"
        lines.append(
            f"  {r['query']:<46} {r['gt_class']:<14} "
            f"{r['avlmaps_pick']:<16}{ma}  "
            f"{r['textbridge_pick']:<18}{mt}"
        )
    lines.append("")
    lines.append(
        f"  AVLMaps    Top-1 = {result['avlmaps_top1']*100:5.1f}%   "
        f"({sum(r['ok_avlmaps'] for r in result['rows'])} / {result['n']})"
    )
    lines.append(
        f"  TextBridge Top-1 = {result['textbridge_top1']*100:5.1f}%   "
        f"({sum(r['ok_textbridge'] for r in result['rows'])} / {result['n']})"
    )
    lines.append("=" * 96)
    return "\n".join(lines)


if __name__ == "__main__":
    res = evaluate()
    print(format_report(res))
