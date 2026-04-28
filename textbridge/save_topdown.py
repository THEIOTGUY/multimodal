"""Save one top-down PNG per query into output/topdown/.

For each instruction in `QUERIES`, render three panes side-by-side:
  1. AVLMaps heatmap + pick (red)
  2. TextBridge heatmap + pick (green)
  3. Real RGB top-down: rooms + furniture-icon for each source, no heatmap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from textbridge.animate import START_RC, _draw_floor_plan, _draw_sources, _path
from textbridge.captioner import CAPTIONS_PATH
from textbridge.query import QUERIES, heatmap, rank_sources
from textbridge.scene import attach_captions


OUT_DIR = Path("output/topdown")


# Per-source short text label for the "real" top-down pane (font-safe).
SID_ICON = {
    "S1": "dog",
    "S2": "vacuum",
    "S3": "glass",
    "S4": "washer",
    "S5": "keyboard",
    "S6": "baby",
    "S7": "door",
    "S8": "toothbrush",
    "S9": "sink",
}


def _slug(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")


def _draw_path(ax, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    pts = _path(start, end, n=80)
    ax.plot(pts[:, 1], pts[:, 0], color="#0033cc", lw=2.0, alpha=0.85, zorder=20)
    ax.scatter([start[1]], [start[0]], c="#0033cc", marker="s", s=70,
               edgecolors="white", linewidths=1.2, zorder=30, label="start")
    ax.scatter([end[1]], [end[0]], c="#0033cc", marker="^", s=180,
               edgecolors="white", linewidths=1.5, zorder=30, label="robot")


def _draw_real_rgb_pane(ax, sources, gt: str) -> None:
    """RGB top-down: colored rooms + emoji icon per source. No heatmap."""
    _draw_floor_plan(ax)
    for s in sources:
        icon = SID_ICON.get(s.sid, "?")
        is_gt = s.sid == gt
        edge = "#1a8000" if is_gt else "#222222"
        face = "#dff5d5" if is_gt else "#ffffff"
        ax.scatter([s.col], [s.row], facecolors=face, edgecolors=edge,
                   s=520, linewidths=1.6, zorder=14)
        ax.text(s.col, s.row, icon, ha="center", va="center",
                fontsize=8, fontweight="bold", color="black", zorder=15)
        ax.annotate(s.sid, (s.col, s.row), xytext=(16, 10),
                    textcoords="offset points", fontsize=7,
                    color="black", zorder=16)
    ax.scatter([START_RC[1]], [START_RC[0]], c="#0033cc", marker="s", s=70,
               edgecolors="white", linewidths=1.2, zorder=20)
    ax.annotate("start", (START_RC[1], START_RC[0]), xytext=(8, -2),
                textcoords="offset points", fontsize=7, color="#0033cc",
                zorder=20)


def render_one(query: str, gt: str, sources, sid2src, out_path: Path) -> None:
    h_avl    = heatmap(query, sources, pipeline="avlmaps")
    h_bridge = heatmap(query, sources, pipeline="textbridge")
    avl_pick,    avl_cos    = rank_sources(query, sources, pipeline="avlmaps")[0]
    bridge_pick, bridge_cos = rank_sources(query, sources, pipeline="textbridge")[0]

    fig, (ax_l, ax_r, ax_real) = plt.subplots(1, 3, figsize=(19.5, 6.0))
    fig.suptitle(
        f'instruction: "{query}"   |   GT: {gt} ({sid2src[gt].class_label})',
        fontsize=12, fontweight="bold",
    )

    _draw_floor_plan(ax_l)
    _draw_floor_plan(ax_r)

    ax_l.imshow(np.ma.masked_less(h_avl, 0.10), cmap="Reds",
                alpha=0.55, vmin=0, vmax=1, origin="upper",
                interpolation="bilinear", zorder=2)
    ax_r.imshow(np.ma.masked_less(h_bridge, 0.10), cmap="Greens",
                alpha=0.55, vmin=0, vmax=1, origin="upper",
                interpolation="bilinear", zorder=2)

    _draw_sources(ax_l, sources, gt, avl_pick.sid)
    _draw_sources(ax_r, sources, gt, bridge_pick.sid)

    _draw_path(ax_l, START_RC, (avl_pick.row,    avl_pick.col))
    _draw_path(ax_r, START_RC, (bridge_pick.row, bridge_pick.col))

    _draw_real_rgb_pane(ax_real, sources, gt)

    ok_a = avl_pick.sid    == gt
    ok_b = bridge_pick.sid == gt
    ax_l.set_title(
        f'AVLMaps pick: {avl_pick.sid} ({avl_pick.class_label})  '
        f'cos={avl_cos:+.3f}  ' + ("✓" if ok_a else "✗"),
        color="#0b6b0b" if ok_a else "#b01010", fontsize=10,
    )
    ax_r.set_title(
        f'TextBridge pick: {bridge_pick.sid} ({bridge_pick.class_label})  '
        f'cos={bridge_cos:+.3f}  ' + ("✓" if ok_b else "✗"),
        color="#0b6b0b" if ok_b else "#b01010", fontsize=10,
    )
    ax_real.set_title("Real RGB top-down (rooms + sources)", fontsize=10)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captions = json.loads(CAPTIONS_PATH.read_text())
    sources = attach_captions(captions)
    sid2src = {s.sid: s for s in sources}

    for query, gt in QUERIES:
        out = OUT_DIR / f"{_slug(query)}.png"
        render_one(query, gt, sources, sid2src, out)
        print(f"[topdown] {query!r} -> {out.resolve()}")


if __name__ == "__main__":
    main()
