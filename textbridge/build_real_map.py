"""Build VLMap + AreaMap + SoundMap for 5LpN3gDmAk7_1 without going through
the full `AVLMap` class (which drags in `third_party.SuperGluePretrainedNetwork`
via VisualMap, and we don't need image-query localization).

Imports each map directly from its module so `avlmaps/map/__init__.py`'s
VisualMap import never runs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Direct imports to bypass avlmaps.map.__init__.py (which imports VisualMap →
# third_party.SuperGluePretrainedNetwork).
from avlmaps.map.vlmap import VLMap
from avlmaps.map.area_map import AreaMap
from avlmaps.map.sound_map import SoundMap


def build_one(config: DictConfig, scene_dir: Path) -> None:
    print(f"\n[build] scene_dir = {scene_dir}")

    print("[build] VLMap.create_map (LSeg + voxel fusion)")
    vlmap = VLMap(config.map_config, data_dir=str(scene_dir))
    vlmap.create_map(scene_dir)

    print("[build] AreaMap.create_map")
    area_map = AreaMap(str(scene_dir))
    area_map.create_map(scene_dir)

    print("[build] SoundMap.create_sound_map (AudioCLIP encoding of inserted clips)")
    sound_map = SoundMap(
        str(scene_dir),
        config.sound_config,
        config.sound_data_collect_params,
        is_ambiguous=False,
        is_real=False,
    )
    sound_map.create_sound_map(scene_dir)

    print("[build] done.")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    data_dir = Path(config.data_paths.avlmaps_data_dir) / "avlmaps_dataset"
    scene_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    scene_dir = scene_dirs[config.scene_id]
    build_one(config, scene_dir)


if __name__ == "__main__":
    main()
