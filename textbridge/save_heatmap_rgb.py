"""For each query save heatmap | RGB top-down side-by-side PNGs.

Left pane: TextBridge similarity heatmap (where the system thinks the
matching source is).
Right pane: RGB top-down — colored rooms + all 8 sources + L-shaped robot
path from the hallway start to the picked source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from textbridge.animate import START_RC, _draw_floor_plan, _draw_sources, _path
from textbridge.captioner import CAPTIONS_PATH
from textbridge.query import QUERIES, heatmap, rank_sources
from textbridge.save_topdown import _slug
from textbridge.scene import GRID_H, GRID_W, ROOMS, attach_captions


OUT_DIR = Path("output/heatmap_rgb")


def _draw_path(ax, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    pts = _path(start, end, n=80)
    ax.plot(pts[:, 1], pts[:, 0], color="#0033cc", lw=2.0, alpha=0.85, zorder=20)
    ax.scatter([start[1]], [start[0]], c="#0033cc", marker="s", s=70,
               edgecolors="white", linewidths=1.2, zorder=30)
    ax.scatter([end[1]], [end[0]], c="#0033cc", marker="^", s=180,
               edgecolors="white", linewidths=1.5, zorder=30)


def _draw_heatmap_pane(ax, h: np.ndarray) -> None:
    ax.imshow(np.full((GRID_H, GRID_W, 3), 1.0), origin="upper")
    for name, (r0, r1, c0, c1), _ in ROOMS:
        ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0],
                color="#bbbbbb", lw=1.0, zorder=1)
        if (r1 - r0) >= 8 and (c1 - c0) >= 8:
            ax.text((c0 + c1) / 2, (r0 + r1) / 2, name.replace("_", " "),
                    ha="center", va="center", fontsize=8, alpha=0.4,
                    color="black")
    im = ax.imshow(h, cmap="magma", vmin=0, vmax=1, origin="upper",
                   interpolation="bilinear", alpha=0.92, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1, GRID_W); ax.set_ylim(GRID_H, -1)
    plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02, label="similarity")


def render_one(query: str, gt: str, sources, sid2src, out_path: Path) -> None:
    h_bridge = heatmap(query, sources, pipeline="textbridge")
    bridge_pick, bridge_cos = rank_sources(query, sources,
                                           pipeline="textbridge")[0]

    fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.2))
    ok = bridge_pick.sid == gt
    fig.suptitle(
        f'instruction: "{query}"   |   GT: {gt} ({sid2src[gt].class_label})   |   '
        f'TextBridge pick: {bridge_pick.sid} ({bridge_pick.class_label})  '
        f'cos={bridge_cos:+.3f}  ' + ("✓" if ok else "✗"),
        fontsize=11, fontweight="bold",
        color="#0b6b0b" if ok else "#b01010",
    )

    ax_h.set_title("TextBridge heatmap (SBERT cos: query vs Qwen captions)",
                   fontsize=10)
    _draw_heatmap_pane(ax_h, h_bridge)

    ax_r.set_title("RGB top-down (rooms + sources + robot path)", fontsize=10)
    _draw_floor_plan(ax_r)
    _draw_sources(ax_r, sources, gt, bridge_pick.sid)
    _draw_path(ax_r, START_RC, (bridge_pick.row, bridge_pick.col))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
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
        print(f"[heatmap_rgb] {query!r} -> {out.resolve()}")


if __name__ == "__main__":
    main()
