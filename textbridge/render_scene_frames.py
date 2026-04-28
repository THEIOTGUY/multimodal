"""Minimal frame-renderer for a single AVLMaps scene.

Reads the existing `<scene>_<floor>/poses.txt` produced by AVLMaps and renders
rgb/depth/semantic frames at each pose using habitat-sim against the
Matterport3D `.glb`. Writes them to `<scene>_<floor>/{rgb,depth,semantic}/`,
which is exactly what `VLMap.create_map` expects to find.

Skips the audio_video sub-sequences that the original `generate_dataset.py`
also processes — those need ESC-50 audio insertion which isn't needed for
the top-down RGB map.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import habitat_sim
import numpy as np
from tqdm import tqdm

from avlmaps.utils.habitat_utils import get_obj2cls_dict, make_cfg, save_obs


def render_scene(scene_data_dir: Path, scene_glb: Path,
                 width: int = 1080, height: int = 720,
                 camera_height: float = 1.5) -> None:
    poses = np.loadtxt(scene_data_dir / "poses.txt")
    if poses.ndim == 1:
        poses = poses[None]

    sim_setting = {
        "scene": str(scene_glb),
        "default_agent": 0,
        "sensor_height": camera_height,
        "color_sensor": True,
        "depth_sensor": True,
        "semantic_sensor": True,
        "move_forward": 0.1,
        "turn_left": 5,
        "turn_right": 5,
        "width": width,
        "height": height,
        "enable_physics": False,
        "seed": 42,
    }

    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    print(f"[render] scene={scene_glb}")
    print(f"[render] writing to {scene_data_dir}/{{rgb,depth,semantic}}/  ({len(poses)} frames)")
    cfg = make_cfg(sim_setting)
    sim = habitat_sim.Simulator(cfg)
    obj2cls = get_obj2cls_dict(sim)
    sim.initialize_agent(sim_setting["default_agent"])

    pbar = tqdm(poses)
    for pose_i, pose in enumerate(pbar):
        pbar.set_description(f"Frame {pose_i:06}")
        st = habitat_sim.AgentState()
        st.position = pose[:3]
        st.rotation = pose[3:]
        sim.get_agent(0).set_state(st)
        obs = sim.get_sensor_observations(0)
        save_obs(scene_data_dir, sim_setting, obs, pose_i, obj2cls)

    sim.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-data-dir", type=Path,
                   default=Path("/usershome/cs671_user6/avlmaps_data/avlmaps_dataset/5LpN3gDmAk7_1"))
    p.add_argument("--scene-glb", type=Path,
                   default=Path("/usershome/cs671_user6/mp3d_data/extracted/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb"))
    args = p.parse_args()
    render_scene(args.scene_data_dir, args.scene_glb)


if __name__ == "__main__":
    main()
