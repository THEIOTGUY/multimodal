"""Real-scene Top-1 retrieval benchmark on the AVLMaps audio dataset.

Reads the `range_and_audio_meta_level_*.txt` files under each scene's
audio_video directory to recover (scene, sequence, frame_range, category,
wav_path) tuples — i.e. exactly which ESC-50 clips were placed where by
`assign_sound_to_video_batch`.

For each unique wav, we compute a Qwen2.5-Omni-7B caption (cached). For
each (query, ground-truth-category) pair we then check whether each
pipeline picks the right *category* among the categories actually placed
in the scene at this difficulty level.

  * AVLMaps baseline: source-side string is the ESC-50 class label.
  * TextBridge:       source-side string is the Qwen caption.

Both pipelines share Sentence-BERT (all-mpnet-base-v2). The query bank is
expanded by reusing the abstract phrasings from `benchmark_100.py` for
every category that appears in the scene at this level.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from textbridge.encoder import embed
from textbridge.queries_30cat import QUERIES_30CAT


DEFAULT_DATASET_ROOT = Path(
    "/usershome/cs671_user6/avlmaps_data/avlmaps_dataset"
)
DEFAULT_LEVELS = ["level_1", "level_2", "level_3"]
CAPTIONS_REAL_PATH = Path(__file__).resolve().parent / "captions_real.json"


def parse_placement(scene_dir: Path, level: str
                    ) -> List[Dict]:
    """Read every range_and_audio_meta_<level>.txt under this scene.

    Returns one dict per audio event:
      { scene, seq, frame_start, frame_end, category, wav_path }
    Category is normalised to ESC-50 underscore form (e.g. "glass_breaking").
    """
    rows = []
    av_dir = scene_dir / "audio_video"
    for seq_dir in sorted(av_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        meta = seq_dir / f"range_and_audio_meta_{level}.txt"
        if not meta.exists():
            continue
        for line in meta.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 5)
            if len(parts) != 6:
                continue
            f0, f1, t0, t1, cat, wav = parts
            rows.append({
                "scene":       scene_dir.name,
                "seq":         seq_dir.name,
                "frame_start": int(f0),
                "frame_end":   int(f1),
                "category":    cat.replace(" ", "_"),
                "wav_path":    wav,
            })
    return rows


def load_or_build_real_captions(wav_paths: List[str]) -> Dict[str, str]:
    """Caption every unique wav with Qwen2.5-Omni-7B and persist to JSON."""
    cache: Dict[str, str] = {}
    if CAPTIONS_REAL_PATH.exists():
        cache = json.loads(CAPTIONS_REAL_PATH.read_text())
    todo = [w for w in wav_paths if w not in cache]
    if not todo:
        return cache

    # Lazy: only import qwen if needed.
    print(f"[caption] {len(todo)} new wavs need Qwen captions")
    import torch
    from transformers import (
        Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
    )
    from textbridge.captioner import _qwen_caption_one

    print("[caption] loading Qwen2.5-Omni-7B…")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.float16, device_map="auto",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    for i, wav in enumerate(todo, 1):
        try:
            cap = _qwen_caption_one(model, processor, Path(wav))
        except Exception as e:
            print(f"  [{i}/{len(todo)}] FAILED on {Path(wav).name}: {e}")
            cap = f"(captioning failed: {type(e).__name__})"
        cache[wav] = cap
        print(f"  [{i}/{len(todo)}] {Path(wav).name} -> {cap}")
        CAPTIONS_REAL_PATH.write_text(json.dumps(cache, indent=2))
    return cache


def _query_bank_for_categories(categories: List[str]
                               ) -> List[Tuple[str, str]]:
    """Filter the 300-query bank to queries whose GT category is actually
    placed in this scene. Keeps the comparison fair (the source pool is the
    placed categories; the query pool is queries asking for those)."""
    cats = set(categories)
    return [(q, gt) for q, gt in QUERIES_30CAT if gt in cats]


def _rank_by_strings(query: str, src_strings: List[str]) -> List[int]:
    src_emb = embed(src_strings)
    q_emb = embed([query])[0]
    sims = src_emb @ q_emb
    order = np.argsort(-sims)
    return order.tolist()


def evaluate_scene(scene_dir: Path, level: str,
                   captions: Dict[str, str]) -> Dict:
    placements = parse_placement(scene_dir, level)
    if not placements:
        return {}
    cat_to_wav: Dict[str, str] = {}
    cat_order: List[str] = []
    for p in placements:
        if p["category"] not in cat_to_wav:
            cat_to_wav[p["category"]] = p["wav_path"]
            cat_order.append(p["category"])

    src_class_strings   = [c.replace("_", " ") for c in cat_order]
    src_caption_strings = [captions.get(cat_to_wav[c], "") for c in cat_order]
    # Hybrid: class label + caption joined. Gives the SBERT cosine both the
    # canonical category word AND the rich caption descriptors.
    src_hybrid_strings  = [
        f"{cls_s}. {cap_s}" if cap_s else cls_s
        for cls_s, cap_s in zip(src_class_strings, src_caption_strings)
    ]

    queries = _query_bank_for_categories(cat_order)

    cls_hits = cap_hits = hyb_hits = 0
    rows = []
    for q, gt_cat in queries:
        if gt_cat not in cat_to_wav:
            continue
        cls_order = _rank_by_strings(q, src_class_strings)
        cap_order = _rank_by_strings(q, src_caption_strings)
        hyb_order = _rank_by_strings(q, src_hybrid_strings)
        cls_pick = cat_order[cls_order[0]]
        cap_pick = cat_order[cap_order[0]]
        hyb_pick = cat_order[hyb_order[0]]
        ok_cls = cls_pick == gt_cat
        ok_cap = cap_pick == gt_cat
        ok_hyb = hyb_pick == gt_cat
        cls_hits += int(ok_cls); cap_hits += int(ok_cap); hyb_hits += int(ok_hyb)
        rows.append({
            "query": q, "gt_category": gt_cat,
            "avlmaps_pick": cls_pick, "ok_avlmaps": ok_cls,
            "textbridge_pick": cap_pick, "ok_textbridge": ok_cap,
            "hybrid_pick": hyb_pick, "ok_hybrid": ok_hyb,
            "textbridge_caption_used":
                captions.get(cat_to_wav[gt_cat], ""),
        })
    n = len(rows)
    return {
        "scene": scene_dir.name,
        "level": level,
        "n_categories": len(cat_order),
        "categories": cat_order,
        "n_queries": n,
        "avlmaps_top1":   cls_hits / max(n, 1),
        "textbridge_top1": cap_hits / max(n, 1),
        "hybrid_top1":    hyb_hits / max(n, 1),
        "rows": rows,
        "category_caption": {c: captions.get(cat_to_wav[c], "") for c in cat_order},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    p.add_argument("--scenes", nargs="*", default=None)
    args = p.parse_args()

    if args.scenes is None:
        scene_dirs = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    else:
        scene_dirs = [args.root / s for s in args.scenes]

    # Gather placements first to get the union of wavs we need captioned.
    all_wavs = set()
    plan = []
    for sd in scene_dirs:
        for lvl in args.levels:
            placements = parse_placement(sd, lvl)
            if not placements:
                continue
            plan.append((sd, lvl, placements))
            all_wavs.update(p["wav_path"] for p in placements)
    print(f"[bench] scenes scanned: {len(scene_dirs)}, with placements: {len(plan)}")
    print(f"[bench] unique wavs to caption: {len(all_wavs)}")

    captions = load_or_build_real_captions(sorted(all_wavs))

    per_scene = []
    cls_total_hits = cap_total_hits = hyb_total_hits = total_n = 0
    for sd, lvl, _ in plan:
        res = evaluate_scene(sd, lvl, captions)
        if not res:
            continue
        per_scene.append(res)
        n = res["n_queries"]
        cls_total_hits += res["avlmaps_top1"]    * n
        cap_total_hits += res["textbridge_top1"] * n
        hyb_total_hits += res["hybrid_top1"]     * n
        total_n += n
        print(f"  {sd.name:18s} {lvl}  n={n:3d}  "
              f"avlmaps={res['avlmaps_top1']*100:5.1f}%  "
              f"textbridge={res['textbridge_top1']*100:5.1f}%  "
              f"hybrid={res['hybrid_top1']*100:5.1f}%")

    summary = {
        "n_scenes": len(per_scene),
        "n_queries_total": total_n,
        "avlmaps_top1":    (cls_total_hits / total_n) if total_n else 0.0,
        "textbridge_top1": (cap_total_hits / total_n) if total_n else 0.0,
        "hybrid_top1":     (hyb_total_hits / total_n) if total_n else 0.0,
        "per_scene": per_scene,
        "captions_real_path": str(CAPTIONS_REAL_PATH),
    }

    out_local = Path(__file__).resolve().parent / "benchmark_real.json"
    out_local.write_text(json.dumps(summary, indent=2))
    out_run = Path("output/benchmark_real.json")
    out_run.parent.mkdir(parents=True, exist_ok=True)
    out_run.write_text(json.dumps(summary, indent=2))

    print()
    print(f"  TOTAL  n={total_n:4d}  "
          f"avlmaps={summary['avlmaps_top1']*100:5.1f}%  "
          f"textbridge={summary['textbridge_top1']*100:5.1f}%  "
          f"hybrid={summary['hybrid_top1']*100:5.1f}%")
    print(f"\n[bench] wrote -> {out_local}")


if __name__ == "__main__":
    main()
