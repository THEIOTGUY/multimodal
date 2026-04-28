"""Render the TextBridge navigation MP4.

For each instruction we let *both* pipelines decide which sound source to
walk to, then animate a robot navigating from its current cell to the picked
source over a few seconds. Side-by-side panes make the abstract-instruction
wins immediately visible: the right-hand robot reaches the correct room,
the left-hand one walks into the wrong one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

from textbridge.captioner import CAPTIONS_PATH
from textbridge.query import QUERIES, heatmap, rank_sources
from textbridge.scene import GRID_H, GRID_W, ROOMS, attach_captions, room_color_grid


FPS = 15
TRAVEL_S = 1.6     # seconds the robot takes to traverse a leg
HOLD_S = 1.0       # seconds it lingers at the picked source

START_RC = (40, 70)  # hallway centre


# ---------------------------------------------------------------------------
# Path planning: simple two-segment Manhattan walk with cell smoothing.
# ---------------------------------------------------------------------------

def _path(rc0: Tuple[int, int], rc1: Tuple[int, int], n: int) -> np.ndarray:
    """Linear-segment path with `n` waypoints between rc0 and rc1, then a
    second linear segment to add a slight L-shape so the trail is readable."""
    r0, c0 = rc0
    r1, c1 = rc1
    # waypoint via the column of the start row at the destination column
    rm, cm = r0, c1
    half = n // 2
    seg1 = np.stack([
        np.linspace(r0, rm, half, endpoint=False),
        np.linspace(c0, cm, half, endpoint=False),
    ], axis=1)
    seg2 = np.stack([
        np.linspace(rm, r1, n - half),
        np.linspace(cm, c1, n - half),
    ], axis=1)
    return np.concatenate([seg1, seg2], axis=0)


def _build_timeline(picks_per_q: List[Tuple[Tuple[int, int], Tuple[int, int]]]
                    ) -> List[dict]:
    """For every query produce per-frame state for the LEFT and RIGHT robot.

    Each state has the (row, col) the robot is at, plus a flag saying
    whether the robot has arrived (used to flash the heatmap stronger).
    """
    travel_n = int(FPS * TRAVEL_S)
    hold_n   = int(FPS * HOLD_S)

    timeline: List[dict] = []
    cur_left  = np.array(START_RC, dtype=np.float32)
    cur_right = np.array(START_RC, dtype=np.float32)
    for q_i, (left_target, right_target) in enumerate(picks_per_q):
        left_path  = _path(tuple(cur_left.astype(int)),  left_target,  travel_n)
        right_path = _path(tuple(cur_right.astype(int)), right_target, travel_n)
        for f in range(travel_n):
            timeline.append({
                "q_i": q_i, "f": f, "phase": "travel",
                "left_rc":  left_path[f],  "right_rc":  right_path[f],
            })
        for f in range(hold_n):
            timeline.append({
                "q_i": q_i, "f": f, "phase": "hold",
                "left_rc":  left_path[-1],  "right_rc":  right_path[-1],
            })
        cur_left  = left_path[-1]
        cur_right = right_path[-1]
    return timeline


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_floor_plan(ax) -> None:
    img = room_color_grid()
    ax.imshow(img, origin="upper")
    for name, (r0, r1, c0, c1), _ in ROOMS:
        if (r1 - r0) < 8 or (c1 - c0) < 8:
            continue
        ax.text((c0 + c1) / 2, (r0 + r1) / 2, name.replace("_", " "),
                ha="center", va="center", fontsize=8, alpha=0.55,
                color="black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1, GRID_W); ax.set_ylim(GRID_H, -1)


def _draw_sources(ax, sources, gt_sid, pick_sid) -> None:
    for s in sources:
        if s.sid == gt_sid:
            color, marker, size = "#1a8000", "*", 220
        elif s.sid == pick_sid:
            color, marker, size = "#c41010", "X", 140
        else:
            color, marker, size = "#222222", "o", 35
        ax.scatter([s.col], [s.row], c=color, marker=marker, s=size,
                   edgecolors="white", linewidths=1.0, zorder=10)
        ax.annotate(s.sid, (s.col, s.row), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="black",
                    zorder=11)


def render(out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    captions = json.loads(CAPTIONS_PATH.read_text())
    sources = attach_captions(captions)
    sid2src = {s.sid: s for s in sources}

    # Pre-compute heatmaps + picks per query.
    cache = []
    picks_rc = []
    for query, gt in QUERIES:
        h_avl    = heatmap(query, sources, pipeline="avlmaps")
        h_bridge = heatmap(query, sources, pipeline="textbridge")
        avl_pick    = rank_sources(query, sources, pipeline="avlmaps")[0]
        bridge_pick = rank_sources(query, sources, pipeline="textbridge")[0]
        cache.append({
            "query": query, "gt": gt,
            "h_avl": h_avl, "h_bridge": h_bridge,
            "avl_pick": avl_pick[0], "avl_cos": avl_pick[1],
            "bridge_pick": bridge_pick[0], "bridge_cos": bridge_pick[1],
        })
        picks_rc.append((
            (avl_pick[0].row,    avl_pick[0].col),
            (bridge_pick[0].row, bridge_pick[0].col),
        ))
    timeline = _build_timeline(picks_rc)

    # ----------------------------- figure layout -----------------------------
    fig = plt.figure(figsize=(15.0, 9.5))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[0.6, 5.2, 0.5, 2.7], hspace=0.30, wspace=0.10,
        left=0.04, right=0.97, top=0.93, bottom=0.04,
    )
    ax_query  = fig.add_subplot(gs[0, :]); ax_query.axis("off")
    ax_left   = fig.add_subplot(gs[1, 0])
    ax_right  = fig.add_subplot(gs[1, 1])
    ax_pick   = fig.add_subplot(gs[2, :]); ax_pick.axis("off")
    ax_legend = fig.add_subplot(gs[3, :]); ax_legend.axis("off")

    fig.suptitle(
        "TextBridge — robot navigation under abstract instructions  "
        "(left: AVLMaps cross-modal cosine,  "
        "right: ours, audio→Qwen2.5-Omni-7B caption→Sentence-BERT)",
        fontsize=12, fontweight="bold",
    )

    _draw_floor_plan(ax_left)
    _draw_floor_plan(ax_right)

    left_im  = ax_left.imshow(np.zeros((GRID_H, GRID_W)), cmap="Reds",
                              alpha=0.55, vmin=0, vmax=1, origin="upper",
                              interpolation="bilinear", zorder=2)
    right_im = ax_right.imshow(np.zeros((GRID_H, GRID_W)), cmap="Greens",
                               alpha=0.55, vmin=0, vmax=1, origin="upper",
                               interpolation="bilinear", zorder=2)

    # Robot dot + trail
    left_trail,  = ax_left.plot([], [], color="#0033cc", lw=2.0, zorder=20,
                                 alpha=0.85)
    right_trail, = ax_right.plot([], [], color="#0033cc", lw=2.0, zorder=20,
                                  alpha=0.85)
    left_robot   = ax_left.scatter([], [], c="#0033cc", s=180, marker="^",
                                    edgecolors="white", linewidths=1.5,
                                    zorder=30)
    right_robot  = ax_right.scatter([], [], c="#0033cc", s=180, marker="^",
                                     edgecolors="white", linewidths=1.5,
                                     zorder=30)

    query_text = ax_query.text(0.5, 0.5, "", ha="center", va="center",
                               fontsize=14, fontweight="bold", color="#222222")
    pick_text_left  = ax_pick.text(0.25, 0.5, "", ha="center", va="center",
                                   fontsize=10)
    pick_text_right = ax_pick.text(0.75, 0.5, "", ha="center", va="center",
                                   fontsize=10)

    legend_lines = ["Sources (placed in semantically appropriate rooms; ★=GT, ✕=robot pick, ▲=robot):"]
    for s in sources:
        legend_lines.append(
            f"  {s.sid}  [{s.class_label:16s}]  ({s.room:11s})  "
            f"Qwen-caption: “{s.caption}”"
        )
    ax_legend.text(0.0, 1.0, "\n".join(legend_lines),
                   fontsize=8, va="top", family="monospace")

    # Per-frame trails accumulated so the robot leaves a path.
    trail_left:  List[Tuple[float, float]] = []
    trail_right: List[Tuple[float, float]] = []
    last_q_i = [-1]

    def update(idx: int):
        st = timeline[idx]
        c = cache[st["q_i"]]

        # New query -> reset trail.
        if st["q_i"] != last_q_i[0]:
            trail_left.clear()
            trail_right.clear()
            last_q_i[0] = st["q_i"]
            # update static overlays
            left_im.set_data(np.ma.masked_less(c["h_avl"],    0.10))
            right_im.set_data(np.ma.masked_less(c["h_bridge"], 0.10))
            for ax in (ax_left, ax_right):
                # remove per-source markers from previous query
                for art in list(ax.collections):
                    if art is left_robot or art is right_robot:
                        continue
                    art.remove()
                for art in list(ax.texts):
                    if art.get_fontsize() == 7:
                        art.remove()
            _draw_sources(ax_left,  sources, c["gt"], c["avl_pick"].sid)
            _draw_sources(ax_right, sources, c["gt"], c["bridge_pick"].sid)
            gt_class = sid2src[c["gt"]].class_label
            query_text.set_text(
                f'instruction: “{c["query"]}”      '
                f'ground-truth source: {c["gt"]} ({gt_class})'
            )
            ok_a = c["avl_pick"].sid    == c["gt"]
            ok_b = c["bridge_pick"].sid == c["gt"]
            pick_text_left.set_text(
                f'AVLMaps pick: {c["avl_pick"].sid} '
                f'({c["avl_pick"].class_label})  cos = {c["avl_cos"]:+.3f}    '
                + ("✓ navigating to correct source" if ok_a
                   else "✗ navigating to WRONG source")
            )
            pick_text_left.set_color("#0b6b0b" if ok_a else "#b01010")
            pick_text_right.set_text(
                f'TextBridge pick: {c["bridge_pick"].sid} '
                f'({c["bridge_pick"].class_label})  '
                f'cos = {c["bridge_cos"]:+.3f}    '
                + ("✓ navigating to correct source" if ok_b
                   else "✗ navigating to WRONG source")
            )
            pick_text_right.set_color("#0b6b0b" if ok_b else "#b01010")

        lr = st["left_rc"]
        rr = st["right_rc"]
        trail_left.append((lr[1],  lr[0]))
        trail_right.append((rr[1], rr[0]))
        lt = np.array(trail_left)
        rt = np.array(trail_right)
        left_trail.set_data(lt[:, 0],  lt[:, 1])
        right_trail.set_data(rt[:, 0], rt[:, 1])
        left_robot.set_offsets([[lr[1],  lr[0]]])
        right_robot.set_offsets([[rr[1], rr[0]]])
        return (left_im, right_im, left_trail, right_trail,
                left_robot, right_robot,
                query_text, pick_text_left, pick_text_right)

    anim = FuncAnimation(fig, update, frames=len(timeline),
                         interval=1000.0 / FPS, blit=False)
    writer = FFMpegWriter(
        fps=FPS, codec="libx264", bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(out_path.as_posix(), writer=writer, dpi=85)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = render("output/textbridge_demo.mp4")
    print(f"wrote {out}")
