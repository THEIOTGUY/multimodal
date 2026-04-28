"""Synthetic 2D scene with 8 real ESC-50 audio sources placed in semantically
appropriate rooms.

Each source is described two ways:
  * `class_label`  — what AVLMaps' AudioCLIP / CLAP audio encoder effectively
                     gives you: a short, ESC-50-class-level token.
  * `caption`      — what an audio LLM (Qwen2.5-Omni-7B) outputs when asked
                     to describe the same WAV file in natural language.

The captions field is filled in by `captioner.py` at demo time. We persist
them to JSON so the MP4 render does not need the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Floor plan — 12 m x 8 m grid, 0.1 m cells -> 80 x 120
GRID_H = 80
GRID_W = 120

ROOMS = [
    ("kitchen",      (45, 80,   0,  45), (0.95, 0.85, 0.55)),
    ("living_room",  ( 0, 45,   0,  60), (0.70, 0.85, 0.95)),
    ("hallway",      (30, 50,  60,  85), (0.85, 0.85, 0.85)),
    ("bedroom",      ( 0, 45,  85, 120), (0.85, 0.75, 0.95)),
    ("bathroom",     (45, 80,  85, 120), (0.75, 0.95, 0.80)),
]


def room_color_grid() -> np.ndarray:
    img = np.full((GRID_H, GRID_W, 3), 1.0)
    for _, (r0, r1, c0, c1), color in ROOMS:
        img[r0:r1, c0:c1] = color
    return img


@dataclass
class SoundSource:
    """One real ESC-50 wav file mounted in one room."""
    sid: str
    wav_relpath: str          # path under ESC-50-master/audio
    class_label: str          # AVLMaps-style short token
    room: str
    row: int
    col: int
    caption: Optional[str] = None  # filled in by Qwen2.5-Omni-7B

    @property
    def wav_path(self) -> Path:
        return Path("/usershome/cs671_user6/ESC-50-master/audio") / self.wav_relpath


# 8 sources spanning all five rooms. Picked from ESC-50 fold 1.
SOURCES: List[SoundSource] = [
    SoundSource("S1", "1-110389-A-0.wav",   "dog",            "living_room", 18, 25),
    SoundSource("S2", "1-100210-A-36.wav",  "vacuum_cleaner", "living_room", 30, 50),
    SoundSource("S3", "1-20133-A-39.wav",   "glass_breaking", "kitchen",     58, 18),
    SoundSource("S4", "1-21896-A-35.wav",   "washing_machine","kitchen",     70,  8),
    SoundSource("S5", "1-137-A-32.wav",     "keyboard_typing","bedroom",     15,100),
    SoundSource("S6", "1-187207-A-20.wav",  "crying_baby",    "bedroom",     30,108),
    SoundSource("S7", "1-101336-A-30.wav",  "door_wood_knock","hallway",     38, 75),
    SoundSource("S8", "1-17092-A-27.wav",   "brushing_teeth", "bathroom",    65,105),
    SoundSource("S9", "1-118559-A-17.wav",  "pouring_water",  "bathroom",    55,115),
]


# Robot starting pose: middle of the hallway.
ROBOT_HOME = (40, 72)


def attach_captions(captions_by_sid: dict) -> List[SoundSource]:
    """Return a copy of SOURCES with the `caption` field populated."""
    out = []
    for s in SOURCES:
        c = captions_by_sid.get(s.sid)
        out.append(SoundSource(
            sid=s.sid, wav_relpath=s.wav_relpath, class_label=s.class_label,
            room=s.room, row=s.row, col=s.col, caption=c,
        ))
    return out
