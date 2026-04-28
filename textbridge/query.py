"""Per-pipeline query heatmaps and ranking.

Both pipelines build a 2D heatmap over the floor plan by:
  1. Embedding the user's natural-language instruction with Sentence-BERT.
  2. Computing cosine similarity to every sound-source embedding.
  3. Painting a Gaussian blob around each source weighted by that similarity.

The only thing that differs is *what* gets embedded for the source side:
class label (AVLMaps) vs Qwen2.5-Omni-7B caption (TextBridge).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from textbridge.encoder import embed
from textbridge.scene import GRID_H, GRID_W, SoundSource


# Set of language instructions used for the demo. Most are deliberately
# *abstract* / intent-level: AVLMaps' class-token cosine sees no overlap, but
# Qwen's caption usually mentions enough context to make the right cell light
# up under text-text retrieval.
QUERIES: List[Tuple[str, str]] = [
    # (intent_query, ground_truth_sid)
    ("I want to wash my hands",                 "S9"),  # pouring water (sink)
    ("who is at the front entrance",            "S7"),  # door knock
    ("what child needs comfort right now",      "S6"),  # crying baby
    ("where can I find the active home office", "S5"),  # keyboard typing
    ("which appliance just had a loud accident","S3"),  # glass breaking
    ("who is doing their morning hygiene",      "S8"),  # brushing teeth
    ("where is the pet making noise",           "S1"),  # dog barks
    ("find the small whining motor",            "S2"),  # vacuum cleaner
]


def _source_strings(sources: List[SoundSource], pipeline: str) -> List[str]:
    if pipeline == "avlmaps":
        # AudioCLIP / CLAP encoders project audio close to its class label in
        # the joint embedding space, so the class token is the right proxy.
        return [s.class_label.replace("_", " ") for s in sources]
    if pipeline == "textbridge":
        return [s.caption or s.class_label for s in sources]
    raise ValueError(f"unknown pipeline {pipeline!r}")


def rank_sources(query: str, sources: List[SoundSource], pipeline: str
                 ) -> List[Tuple[SoundSource, float]]:
    """Return [(source, cosine), ...] sorted high-to-low cosine."""
    src_strings = _source_strings(sources, pipeline)
    src_emb = embed(src_strings)
    q_emb   = embed([query])[0]
    sims = src_emb @ q_emb
    order = np.argsort(-sims)
    return [(sources[i], float(sims[i])) for i in order]


def heatmap(query: str, sources: List[SoundSource], pipeline: str,
            sigma_cells: float = 5.0) -> np.ndarray:
    """Build a (GRID_H, GRID_W) heatmap by dropping each source's similarity
    score at its cell and applying a Gaussian blur. Normalised to [0, 1].
    """
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    ranked = rank_sources(query, sources, pipeline)
    # Remap cosines to a positive range so negative-cosine sources still
    # produce a faint blob (avoids visual flicker between frames).
    sims = np.array([s for _, s in ranked])
    sims = (sims - sims.min()) / max(sims.max() - sims.min(), 1e-6)
    # Sharpen so the winner dominates.
    sims = sims ** 3
    src_to_w = {src.sid: w for (src, _), w in zip(ranked, sims)}
    for s in sources:
        w = src_to_w[s.sid]
        if 0 <= s.row < GRID_H and 0 <= s.col < GRID_W:
            grid[s.row, s.col] += w
    grid = gaussian_filter(grid, sigma=sigma_cells)
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid
