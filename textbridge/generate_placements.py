"""Fast placement-only metadata generator.

Skips the expensive parts of `dataset/generate_dataset.py` — no rgb
rendering, no ffmpeg video encode, no ffmpeg audio mux. Only writes the
placement metadata files that the retrieval benchmark needs:

  <seq>/meta.txt                          — frame ranges with audio events
  <seq>/range_and_audio_meta_level_*.txt  — which ESC-50 wav at each range

Operates on every scene under <root>/avlmaps_dataset/ because every scene
already has poses.txt for the main trajectory and per-sub-sequence
poses.txt — no Matterport mesh required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist

from avlmaps.utils.audio_utils import get_level_categories
from avlmaps.utils.esc50_utils import ESC50Meta


DATASET_ROOT_DEFAULT = Path("/usershome/cs671_user6/avlmaps_data/avlmaps_dataset")
ESC50_META_DEFAULT   = Path("/usershome/cs671_user6/ESC-50-master/meta/esc50.csv")
ESC50_AUDIO_DEFAULT  = Path("/usershome/cs671_user6/ESC-50-master/audio")
SOUND_CONFIG_PATH    = Path("/usershome/cs671_user6/.MM2/AVLMaps/config/sound_config/sound_config.yaml")
SOUND_PARAMS_PATH    = Path("/usershome/cs671_user6/.MM2/AVLMaps/config/sound_data_collect_params/sound_collect_default.yaml")
LEVELS_DEFAULT       = ["level_1", "level_2", "level_3"]


def _select_audio_frames(poses: np.ndarray, avoid_pos: np.ndarray,
                         fps: float, min_dist: float = 2.0
                         ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Mirror of `audio_utils.select_audio_frames` but pose-only — no rgb_dir
    listing. Walks the sequence in 5-second windows, accepting a window only
    if every pose in it is more than `min_dist` from any previously-occupied
    position. Returns the accepted (start_frame, end_frame) ranges."""
    seq_pos = poses[:, :3]
    pairs: List[Tuple[int, int]] = []
    l = 0
    while l < seq_pos.shape[0]:
        r = l + int(5 * fps)
        group_pos = seq_pos[l:r]
        if group_pos.shape[0] == 0:
            break
        dists = cdist(group_pos, avoid_pos)
        if np.min(dists) > min_dist:
            r = min(r, seq_pos.shape[0] - 1)
            pairs.append((l, r))
            l = r + int(fps)
            avoid_pos = np.concatenate(
                [avoid_pos, np.unique(group_pos, axis=0)], axis=0,
            )
            continue
        group_min_dists = np.min(dists, axis=1)
        ids = np.where(group_min_dists <= 3.0)[0] + l
        if len(ids) == 0:
            l = l + 1
        else:
            l = int(np.max(ids)) + 1
    return pairs, avoid_pos


def write_meta_txt(scene_dir: Path, fps: float, overwrite: bool = True
                   ) -> None:
    av_dir = scene_dir / "audio_video"
    avoid_pos = np.array([[np.inf, np.inf, np.inf]], dtype=np.float32)
    seq_dirs = sorted([p for p in av_dir.iterdir() if p.is_dir()])
    for seq_dir in seq_dirs:
        meta_p = seq_dir / "meta.txt"
        if meta_p.exists() and not overwrite:
            continue
        poses = np.loadtxt(seq_dir / "poses.txt")
        if poses.ndim == 1:
            poses = poses[None]
        pairs, avoid_pos = _select_audio_frames(poses, avoid_pos, fps=fps)
        meta_p.write_text("\n".join(f"{a},{b}" for a, b in pairs))


def assign_placements_for_scene(scene_dir: Path, scene_idx: int,
                                level: str, level_cats: List[str],
                                cat2path: dict, fps: float, seed_base: int,
                                ) -> None:
    """Replicate the placement-decision part of `assign_sound_to_video_batch`
    deterministically, but skip the ffmpeg mux. Just writes the meta files."""
    av_dir = scene_dir / "audio_video"
    seq_dirs = sorted([p for p in av_dir.iterdir() if p.is_dir()])

    np.random.seed(scene_idx + seed_base)

    unassigned = sorted(set(level_cats))

    for seq_dir in seq_dirs:
        meta_path = seq_dir / "meta.txt"
        if not meta_path.exists():
            continue
        # parse frame ranges
        frame_ranges = []
        for line in meta_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_ranges.append((int(parts[0]), int(parts[1])))

        # mirror the original: prefer unassigned cats, then fall back
        if len(unassigned) > 0:
            order = list(unassigned)
            np.random.shuffle(order)
            n_take = min(len(frame_ranges), len(order))
            ranges_used = frame_ranges[:n_take]
            cats = [order[i] for i in range(n_take)]
        else:
            sel = np.random.choice(len(level_cats), len(frame_ranges)).tolist()
            cats = [level_cats[i] for i in sel]
            ranges_used = frame_ranges

        # pick a wav per chosen cat
        wavs = [np.random.choice(cat2path[c], 1)[0] for c in cats]

        # update unassigned
        unassigned = [c for c in unassigned if c not in set(cats)]

        # write range_and_audio_meta_<level>.txt
        out = seq_dir / f"range_and_audio_meta_{level}.txt"
        lines = []
        for (f0, f1), cat, wav in zip(ranges_used, cats, wavs):
            t0 = f0 / fps
            t1 = f1 / fps
            lines.append(f"{f0},{f1},{t0},{t1},{cat},{wav}")
        out.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DATASET_ROOT_DEFAULT)
    p.add_argument("--esc-meta", type=Path, default=ESC50_META_DEFAULT)
    p.add_argument("--esc-audio", type=Path, default=ESC50_AUDIO_DEFAULT)
    p.add_argument("--levels", nargs="+", default=LEVELS_DEFAULT)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="restrict to these scene dir names (e.g. 5LpN3gDmAk7_1)")
    args = p.parse_args()

    sound_config = OmegaConf.load(SOUND_CONFIG_PATH)
    sound_params = OmegaConf.load(SOUND_PARAMS_PATH)
    fps = float(sound_params.fps)
    seed_base = int(sound_params.seed)

    audio_meta = ESC50Meta(str(args.esc_meta), str(args.esc_audio))
    cat2path = audio_meta.get_category_name_to_path_dict()
    print(f"[placements] ESC-50: {len(cat2path)} categories in fold 1, "
          f"avg {np.mean([len(v) for v in cat2path.values()]):.1f} wavs/cat")

    scene_dirs = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    if args.scenes is not None:
        wanted = set(args.scenes)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]

    for scene_idx, scene_dir in enumerate(scene_dirs):
        print(f"\n[placements] [{scene_idx+1}/{len(scene_dirs)}] {scene_dir.name}")
        write_meta_txt(scene_dir, fps=fps, overwrite=True)
        for lvl in args.levels:
            level_cats = get_level_categories(lvl, sound_config)
            print(f"  level={lvl}  cats={len(level_cats)}")
            assign_placements_for_scene(
                scene_dir, scene_idx, lvl, level_cats, cat2path,
                fps=fps, seed_base=seed_base,
            )

    # quick summary
    n_files = 0; n_events = 0
    for sd in scene_dirs:
        for seq in sorted((sd / "audio_video").iterdir()):
            if not seq.is_dir():
                continue
            for lvl in args.levels:
                p = seq / f"range_and_audio_meta_{lvl}.txt"
                if p.exists():
                    n_files += 1
                    n_events += sum(1 for line in p.read_text().splitlines() if line.strip())
    print(f"\n[placements] wrote {n_files} meta files, {n_events} placements total")


if __name__ == "__main__":
    main()
