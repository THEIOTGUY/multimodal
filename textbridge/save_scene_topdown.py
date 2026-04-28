"""Render a real top-down RGB view of 5LpN3gDmAk7 directly via habitat-sim,
and overlay each audio_video sub-sequence's traversal as candidate audio
source points. No VLMap/LSeg/AudioCLIP needed.

Output: output/topdown_real/5LpN3gDmAk7_1_topdown_with_audio_points.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import habitat_sim
import numpy as np

from avlmaps.utils.habitat_utils import make_cfg


SCENE_DATA_DIR = Path("/usershome/cs671_user6/avlmaps_data/avlmaps_dataset/5LpN3gDmAk7_1")
SCENE_GLB = Path("/usershome/cs671_user6/mp3d_data/extracted/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb")
OUT_PATH = Path("output/topdown_real/5LpN3gDmAk7_1_topdown_with_audio_points.png")


def _floor_y_from_poses(scene_data_dir: Path) -> float:
    poses = np.loadtxt(scene_data_dir / "poses.txt")
    if poses.ndim == 1:
        poses = poses[None]
    return float(np.median(poses[:, 1]))


def _scene_xz_bounds(scene_data_dir: Path, pad: float = 1.0
                     ) -> Tuple[float, float, float, float]:
    """Return (xmin, xmax, zmin, zmax) over all main + audio_video poses."""
    pose_files = [scene_data_dir / "poses.txt"]
    av_dir = scene_data_dir / "audio_video"
    if av_dir.exists():
        pose_files += sorted(av_dir.glob("*/poses.txt"))
    xs, zs = [], []
    for pf in pose_files:
        p = np.loadtxt(pf)
        if p.ndim == 1:
            p = p[None]
        xs.append(p[:, 0]); zs.append(p[:, 2])
    xs = np.concatenate(xs); zs = np.concatenate(zs)
    return (xs.min() - pad, xs.max() + pad, zs.min() - pad, zs.max() + pad)


def _ortho_topdown(sim: habitat_sim.Simulator,
                   floor_y: float, height_above: float = 6.0,
                   img_w: int = 1200, img_h: int = 900) -> np.ndarray:
    """Render a top-down RGB by placing the agent high above the scene,
    looking straight down. Uses the existing color sensor of the sim."""
    cx, _, cz = sim.pathfinder.get_random_navigable_point()
    bounds = sim.pathfinder.get_bounds()
    cx = (bounds[0][0] + bounds[1][0]) / 2.0
    cz = (bounds[0][2] + bounds[1][2]) / 2.0

    st = habitat_sim.AgentState()
    st.position = np.array([cx, floor_y + height_above, cz], dtype=np.float32)
    # Pitch -90 so camera looks straight down.
    qx_neg90 = np.array([np.sin(-np.pi / 4), 0.0, 0.0, np.cos(-np.pi / 4)],
                        dtype=np.float32)  # (x, y, z, w)
    st.rotation = qx_neg90
    sim.get_agent(0).set_state(st)
    obs = sim.get_sensor_observations(0)
    rgb = obs["color_sensor"][:, :, :3]
    return rgb


def _world_xz_to_pixel(x: float, z: float, bounds, img_shape) -> Tuple[int, int]:
    xmin, xmax, zmin, zmax = bounds
    H, W = img_shape[:2]
    u = (x - xmin) / (xmax - xmin) * W
    v = (z - zmin) / (zmax - zmin) * H
    return int(round(u)), int(round(v))


def _topdown_via_pathfinder(sim: habitat_sim.Simulator, floor_y: float,
                            meters_per_pixel: float = 0.05) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Use habitat's pathfinder to draw a clean top-down floor map at
    floor_y. Returns (rgb_image, (xmin, xmax, zmin, zmax)) for the bounds the
    map covers."""
    td = sim.pathfinder.get_topdown_view(meters_per_pixel, floor_y)
    bounds = sim.pathfinder.get_bounds()
    xmin, xmax = bounds[0][0], bounds[1][0]
    zmin, zmax = bounds[0][2], bounds[1][2]
    # td: bool mask, True=navigable. Convert to a soft RGB.
    mask = td.astype(np.uint8)
    rgb = np.full((mask.shape[0], mask.shape[1], 3), 230, dtype=np.uint8)  # off-white
    rgb[mask == 1] = (200, 220, 245)  # navigable: pale blue
    return rgb, (xmin, xmax, zmin, zmax)


def _overlay_points(rgb: np.ndarray, points_xz: List[Tuple[float, float, str]],
                    bounds: Tuple[float, float, float, float]) -> np.ndarray:
    out = rgb.copy()
    H, W = out.shape[:2]
    xmin, xmax, zmin, zmax = bounds
    for x, z, label in points_xz:
        u = int(round((x - xmin) / (xmax - xmin) * W))
        v = int(round((z - zmin) / (zmax - zmin) * H))
        if not (0 <= u < W and 0 <= v < H):
            continue
        cv2.circle(out, (u, v), 8, (0, 0, 200), thickness=-1)
        cv2.circle(out, (u, v), 9, (255, 255, 255), thickness=2)
        cv2.putText(out, label, (u + 10, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def collect_audio_points(scene_data_dir: Path) -> List[Tuple[float, float, str]]:
    """One representative point per audio_video sub-sequence (its first pose)."""
    av_dir = scene_data_dir / "audio_video"
    pts = []
    for seq_dir in sorted(av_dir.iterdir()):
        pf = seq_dir / "poses.txt"
        if not pf.exists():
            continue
        p = np.loadtxt(pf)
        if p.ndim == 1:
            p = p[None]
        # Use the median pose so a single point represents the sequence's
        # audio-event area without being skewed by the start.
        x = float(np.median(p[:, 0]))
        z = float(np.median(p[:, 2]))
        pts.append((x, z, seq_dir.name))
    return pts


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    floor_y = _floor_y_from_poses(SCENE_DATA_DIR)
    print(f"[topdown] floor_y (median pose y): {floor_y:.3f}")

    sim_setting = {
        "scene": str(SCENE_GLB),
        "default_agent": 0,
        "sensor_height": 0.0,
        "color_sensor": True,
        "depth_sensor": False,
        "semantic_sensor": False,
        "move_forward": 0.1,
        "turn_left": 5,
        "turn_right": 5,
        "width": 1200,
        "height": 900,
        "enable_physics": False,
        "seed": 42,
    }
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    cfg = make_cfg(sim_setting)
    sim = habitat_sim.Simulator(cfg)
    sim.initialize_agent(sim_setting["default_agent"])

    # Use pathfinder topdown — clean and cropped to the scene's navmesh.
    rgb, bounds = _topdown_via_pathfinder(sim, floor_y, meters_per_pixel=0.04)
    print(f"[topdown] map shape={rgb.shape}, bounds (xmin,xmax,zmin,zmax)={bounds}")

    points = collect_audio_points(SCENE_DATA_DIR)
    print(f"[topdown] {len(points)} audio_video sequence points")

    overlaid = _overlay_points(rgb, points, bounds)
    bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT_PATH), bgr)
    print(f"[topdown] wrote -> {OUT_PATH.resolve()}")
    sim.close()


if __name__ == "__main__":
    main()
