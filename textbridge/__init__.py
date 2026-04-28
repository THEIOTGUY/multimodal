"""TextBridge — replacing AVLMaps' cross-modal audio encoder with audio→text.

The published AVLMaps (Huang et al., ISER 2023; IJRR 2025) encodes household
sounds with AudioCLIP (later CLAP) and stores their joint audio-text
embeddings in a 3D voxel grid. Language queries match against those audio
embeddings via cross-modal cosine similarity.

We replace that audio encoder entirely. Each wav clip is fed to
Qwen2.5-Omni-7B, which emits a one-sentence description. That sentence is
then re-embedded with a standard text encoder (Sentence-BERT
all-mpnet-base-v2). The map stores a *text* embedding per cell, and queries
become pure text-text retrieval.

Why this beats cross-modal cosine:
  * Closes the modality gap: text-text similarity is monotonically more
    accurate than audio-text similarity in CLAP-style models.
  * Unlocks LLM reasoning: a caption like "a dog panting near a water bowl"
    matches "find the thirsty animal" via natural-language semantics that
    neither AudioCLIP nor CLAP can express in a single vector.

Run:
    PYTHONPATH=. python -m textbridge.run_demo
"""
from textbridge.scene import SoundSource, SOURCES, attach_captions
from textbridge.encoder import embed
from textbridge.query import QUERIES, heatmap, rank_sources
from textbridge.eval import evaluate, format_report

__all__ = [
    "SoundSource", "SOURCES", "attach_captions",
    "embed", "QUERIES", "heatmap", "rank_sources",
    "evaluate", "format_report",
]
