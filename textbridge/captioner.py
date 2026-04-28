"""Audio→text captioner backed by Qwen/Qwen2.5-Omni-7B.

Replaces the AudioCLIP / CLAP audio encoder with an audio LLM that emits a
natural-language description of the wav. The caption is then re-embedded via
a text encoder so the rest of the AVLMaps pipeline becomes pure text-text
retrieval, closing the cross-modal gap.

Usage:
    python -m textbridge.captioner   # writes textbridge/captions.json

Captions are persisted to JSON so the demo MP4 can be re-rendered without
the GPU. Re-run `python -m textbridge.captioner --force` to regenerate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

CAPTIONS_PATH = Path(__file__).resolve().parent / "captions.json"

PROMPT = (
    "You are an audio captioning model. Listen to the clip and describe what "
    "is happening in one rich, descriptive English sentence. Mention the "
    "likely source object, the activity, and any context that would help a "
    "household robot reason about who or what is making the sound."
)


def _load_existing() -> Dict[str, str]:
    if CAPTIONS_PATH.exists():
        return json.loads(CAPTIONS_PATH.read_text())
    return {}


def _save(captions: Dict[str, str]) -> None:
    CAPTIONS_PATH.write_text(json.dumps(captions, indent=2))


def _qwen_caption_one(model, processor, wav_path: Path) -> str:
    """Run Qwen2.5-Omni-7B once on a single wav file and return its caption."""
    from qwen_omni_utils import process_mm_info  # type: ignore
    import torch

    conversation = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": "You describe sounds for a household robot."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": str(wav_path)},
            {"type": "text",  "text": PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            return_audio=False,
        )
    out_ids = gen[:, inputs["input_ids"].shape[1]:]
    text_out = processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    # Take only the first sentence to keep captions tight.
    if "." in text_out:
        text_out = text_out.split(".", 1)[0].strip() + "."
    return text_out


def caption_all(force: bool = False) -> Dict[str, str]:
    """Caption every SoundSource via Qwen2.5-Omni-7B and persist to JSON."""
    from textbridge.scene import SOURCES

    captions = {} if force else _load_existing()
    todo = [s for s in SOURCES if force or s.sid not in captions]
    if not todo:
        print(f"[captioner] all {len(SOURCES)} captions already cached at "
              f"{CAPTIONS_PATH}")
        return captions

    print(f"[captioner] loading Qwen2.5-Omni-7B... (this can take ~3 min)")
    import torch
    from transformers import (
        Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
    )

    device_map = "auto"
    dtype = torch.float16
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=dtype, device_map=device_map,
    )
    model.disable_talker()  # we only need the text head
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    print(f"[captioner] captioning {len(todo)} clips...")
    for i, src in enumerate(todo, 1):
        try:
            cap = _qwen_caption_one(model, processor, src.wav_path)
        except Exception as e:  # pragma: no cover
            print(f"  [{i}/{len(todo)}] {src.sid} ({src.class_label}) FAILED: {e}",
                  file=sys.stderr)
            cap = f"(captioning failed: {type(e).__name__})"
        captions[src.sid] = cap
        print(f"  [{i}/{len(todo)}] {src.sid} {src.class_label:18s} -> {cap}")
        _save(captions)
    return captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-caption even if a cache exists.")
    args = ap.parse_args()
    caps = caption_all(force=args.force)
    print(f"\n[captioner] wrote {len(caps)} captions -> {CAPTIONS_PATH}")


if __name__ == "__main__":
    main()
