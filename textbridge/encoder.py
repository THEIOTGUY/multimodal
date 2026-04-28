"""Sentence-BERT text encoder used by both pipelines.

Both AVLMaps-baseline and TextBridge use the *same* text encoder
(`all-mpnet-base-v2`, 768-d). The only difference between the two pipelines
is what string is fed in:

  * AVLMaps baseline: the ESC-50-style class label ("dog", "microwave"…),
    which is a tight proxy for what AudioCLIP / CLAP audio embeddings live
    near in joint cross-modal space.
  * TextBridge (ours): the natural-language caption Qwen2.5-Omni-7B produced
    from the same wav file.

This isolates the contribution to the audio→text bridge.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

_MODEL_NAME = "all-mpnet-base-v2"


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: List[str]) -> np.ndarray:
    """Return (N, 768) L2-normalised embeddings."""
    m = _encoder()
    arr = m.encode(list(texts), normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(arr, dtype=np.float32)
