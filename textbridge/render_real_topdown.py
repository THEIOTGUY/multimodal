"""Render the AVLMaps-style real RGB top-down for 5LpN3gDmAk7_1, with the
sound-query heatmap overlaid — the visual the AVLMaps repo gif shows.

For each query in `QUERIES_REAL` we:
  1. Build the top-down RGB via VLMap.generate_rgb_topdown_map().
  2. Get a 2D heatmap by projecting AVLMap.index_sound_2d() (AudioCLIP-side)
     and our TextBridge equivalent (Qwen-caption→SBERT cosine).
  3. Save a side-by-side PNG with both heatmaps drawn on the real top-down.

Bypasses AVLMap so we don't need third_party/SuperGlue.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import distance_transform_edt

from avlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from avlmaps.map.sound_map import SoundMap
from avlmaps.map.vlmap import VLMap


OUT_DIR = Path("output/topdown_real")


# Abstract instructions — same set as the synthetic demo plus "wash my hands".
QUERIES_REAL: List[str] = [
    "I want to wash my hands",
    "who is at the front entrance",
    "what child needs comfort right now",
    "where can I find the active home office",
    "which appliance just had a loud accident",
    "who is doing their morning hygiene",
    "where is the pet making noise",
    "find the small whining motor",
]


def _heatmap_2d_from_locations(
    occupied_shape: Tuple[int, int, int],
    locations_list: List[List[np.ndarray]],
    probabilities: np.ndarray,
    dataloader: VLMapsDataloaderHabitat,
    decay_rate: float = 0.01,
) -> np.ndarray:
    dist_map = np.zeros(occupied_shape[:2], dtype=np.float32)
    for loc_i, locations in enumerate(locations_list):
        tmp = np.zeros_like(dist_map)
        for location in locations:
            tf_hab = np.eye(4)
            tf_hab[:3, 3] = location
            dataloader.from_habitat_tf(tf_hab)
            row, col, _ = dataloader.to_full_map_pose()
            if 0 <= row < tmp.shape[0] and 0 <= col < tmp.shape[1]:
                tmp[row, col] = probabilities[loc_i]
        dists = distance_transform_edt(tmp == 0)
        con = probabilities[loc_i]
        spread = con - con * dists * decay_rate
        spread = np.where(spread < 0, 0, spread)
        dist_map += spread
    if dist_map.max() > 0:
        dist_map = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min())
    return dist_map


def _overlay_heatmap_on_rgb(rgb: np.ndarray, heatmap_2d: np.ndarray,
                            cmap=cv2.COLORMAP_INFERNO, alpha: float = 0.55
                            ) -> np.ndarray:
    h_u8 = (np.clip(heatmap_2d, 0, 1) * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h_u8, cmap)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    mask = (heatmap_2d > 0.05)[..., None]
    out = rgb.copy()
    blended = (alpha * h_color + (1 - alpha) * rgb).astype(np.uint8)
    out = np.where(mask, blended, out)
    return out


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.data_paths.avlmaps_data_dir) / "avlmaps_dataset"
    scene_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    scene_dir = scene_dirs[config.scene_id]
    print(f"[render] scene_dir = {scene_dir}")

    vlmap = VLMap(config.map_config, data_dir=str(scene_dir))
    if not vlmap.load_map(str(scene_dir)):
        raise SystemExit("VLMap not found — run textbridge.build_real_map first.")

    sound_map = SoundMap(
        str(scene_dir),
        config.sound_config,
        config.sound_data_collect_params,
        is_ambiguous=False,
        is_real=False,
    )
    sound_map.load_sound_map(str(scene_dir))

    dataloader = VLMapsDataloaderHabitat(str(scene_dir), config.map_config, vlmap)
    vlmap.generate_obstacle_map()
    rgb_topdown = vlmap.get_rgb_topdown_map_cropped()

    rmin, rmax = vlmap.rmin, vlmap.rmax
    cmin, cmax = vlmap.cmin, vlmap.cmax

    for q_i, query in enumerate(QUERIES_REAL):
        print(f"[render] [{q_i+1}/{len(QUERIES_REAL)}] {query!r}")
        probs, locs = sound_map.get_distribution_and_locations(query)
        heat_full = _heatmap_2d_from_locations(
            vlmap.occupied_ids.shape, locs, probs, dataloader, decay_rate=0.01,
        )
        heat = heat_full[rmin : rmax + 1, cmin : cmax + 1]
        # rgb_topdown is uint8 RGB
        overlaid = _overlay_heatmap_on_rgb(rgb_topdown, heat)
        bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
        slug = "".join(c if c.isalnum() else "_" for c in query.lower()).strip("_")
        out = OUT_DIR / f"{slug}.png"
        cv2.imwrite(str(out), bgr)
        print(f"[render]   -> {out}")


if __name__ == "__main__":
    main()
