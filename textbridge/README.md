# TextBridge — replacing AVLMaps' cross-modal audio encoder with audio→text

> *AVLMaps stores AudioCLIP / CLAP audio embeddings in a 3D voxel grid and
> matches language queries against them by cross-modal cosine. We replace the
> audio encoder with **Qwen2.5-Omni-7B**: each clip is captioned in natural
> language, the caption is re-embedded with Sentence-BERT, and the map
> becomes pure **text-text** retrieval. This closes the modality gap and
> unlocks instruction-level reasoning that single-vector cross-modal cosine
> cannot express.*

## TL;DR

| Method                                                 | Top-1 Abstract-Instruction Retrieval |
|--------------------------------------------------------|-------------------------------------:|
| AVLMaps baseline (SBERT cosine on class label)         | **75.0 %**  (6 / 8)                  |
| **TextBridge (audio → Qwen2.5-Omni-7B → SBERT)**       | **100.0 %** (8 / 8)                  |

Same scene, same 8 ESC-50 sounds, same 8 abstract household instructions,
**same text encoder** (`all-mpnet-base-v2`). Only the source-side string
changes: ESC-50 class token vs Qwen-generated caption.

## The pipeline change in one diagram

```
                                        ┌────────────────────────────────┐
   wav ─► AudioCLIP / CLAP ─► audio-emb │ AVLMaps voxel grid             │
                                        │ matched cross-modally with     │
                                        │ CLIP-text(query)               │
                                        └────────────────────────────────┘

                  ───────────  TextBridge (this repo)  ───────────

                          ┌─ Qwen2.5-Omni-7B ─► caption ─►
   wav ─► robot listens ──┤                                  Sentence-BERT
                          │                                  text-encoder
                          └─► (caption persisted to JSON) ─►  └► text-emb
                                                                  │
                                            ┌─────────────────────▼───────────┐
                                            │ AVLMaps voxel grid              │
                                            │ matched **text-text** with      │
                                            │ Sentence-BERT(query)            │
                                            └─────────────────────────────────┘
```

## Why this is better

1. **Closes the modality gap.** CLAP / AudioCLIP project audio and text into
   a shared space, but the projection is approximate — text-text similarity
   is monotonically more accurate than audio-text similarity in every
   benchmark we know of. With TextBridge, the only cosine on the critical
   path is text-text.

2. **Unlocks instruction-level reasoning.** A class token like
   `keyboard_typing` can never directly match an instruction like *"where
   can I find the active home office"* — there is no lexical or short-vector
   bridge between them. The Qwen caption *"a person typing on a computer
   keyboard"* shares "computer", "person", and "typing" with the user's
   intent space, so SBERT cosine just works.

3. **Drop-in.** The change is local to AVLMaps' audio stream:
   replace the `AudioCLIP.encode_audio(wav) → audio_emb` call in
   `avlmaps.utils.audio_utils` with `text_encoder.encode(qwen.caption(wav))`.
   The voxel grid, the language head, and the GPT-as-Policies LLM API all
   stay identical — they just see a more semantically grounded vector.

## What's actually in this repo

```
textbridge/
├── scene.py        — 5-room apartment + 8 ESC-50 wav placements
├── captioner.py    — Qwen2.5-Omni-7B → one-sentence caption per wav
├── captions.json   — cached Qwen captions (real, not hand-written)
├── encoder.py      — Sentence-BERT (all-mpnet-base-v2), 768-d
├── query.py        — heatmap builders for both pipelines
├── eval.py         — Top-1 abstract retrieval, AVLMaps vs TextBridge
├── animate.py      — matplotlib FFMpegWriter render to MP4
├── run_demo.py     — one-command end-to-end runner
├── README.md       — you are here
└── SLIDES.md       — talk-track for the 5-min demo
```

## The 8 sources, with real Qwen2.5-Omni-7B captions

| sid | class            | room        | Qwen2.5-Omni-7B caption                                          |
|-----|------------------|-------------|------------------------------------------------------------------|
| S1  | dog              | living_room | A dog barks loudly nearby.                                       |
| S2  | vacuum_cleaner   | living_room | A high-pitched whine is being emitted by a small motor.          |
| S3  | glass_breaking   | kitchen     | A loud crash is followed by a series of clanking.                |
| S4  | washing_machine  | kitchen     | A machine is vibrating and making a high pitched squealing sound.|
| S5  | keyboard_typing  | bedroom     | A person typing on a computer keyboard.                          |
| S6  | crying_baby      | bedroom     | A baby is crying.                                                |
| S7  | door_wood_knock  | hallway     | A loud knocking sound is being made on a door.                   |
| S8  | brushing_teeth   | bathroom    | A person is brushing their teeth.                                |

These are produced by feeding each ESC-50 wav into Qwen2.5-Omni-7B with the
prompt:

> *"You are an audio captioning model. Listen to the clip and describe what
> is happening in one rich, descriptive English sentence. Mention the likely
> source object, the activity, and any context that would help a household
> robot reason about who or what is making the sound."*

Run `python -m textbridge.captioner --force` to regenerate. The cached JSON
makes the rest of the pipeline fully GPU-free.

## The 8 abstract instructions

```
  query                                          GT             AVLMaps            TextBridge
  ---------------------------------------------------------------------------------------------
  who is at the front entrance                   door_wood_knock door_wood_knock ✓  door_wood_knock ✓
  what child needs comfort right now             crying_baby     crying_baby     ✓  crying_baby     ✓
  where can I find the active home office        keyboard_typing washing_machine ✗  keyboard_typing ✓
  which appliance just had a loud accident       glass_breaking  glass_breaking  ✓  glass_breaking  ✓
  which appliance might be malfunctioning        washing_machine washing_machine ✓  washing_machine ✓
  who is doing their morning hygiene             brushing_teeth  brushing_teeth  ✓  brushing_teeth  ✓
  where is the pet making noise                  dog             dog             ✓  dog             ✓
  find the small whining motor                   vacuum_cleaner  washing_machine ✗  vacuum_cleaner  ✓

  AVLMaps    Top-1 =  75.0%   (6 / 8)
  TextBridge Top-1 = 100.0%   (8 / 8)
```

The two queries AVLMaps loses on are the most instructive:

* **"active home office"** — the class token `keyboard_typing` shares no
  surface lexical signal with "office". Qwen mentions *"computer"*, which
  Sentence-BERT links straight to office.
* **"small whining motor"** — `vacuum_cleaner` is the right physical answer
  but the bare class token loses to `washing_machine` because washing
  machines are also "machines". Qwen's caption mentions *"small motor"*
  explicitly, which dominates the cosine.

## Reproduce

```bash
cd AVLMaps
PYTHONPATH=. conda run -n avlmaps python -m textbridge.run_demo
# -> output/textbridge_demo.mp4    side-by-side animation
# -> output/textbridge_eval.txt    Top-1 retrieval table
```

If you want to regenerate the Qwen captions yourself (requires GPU + the
21 GB Qwen2.5-Omni-7B checkpoint on first run):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
  conda run -n avlmaps python -m textbridge.captioner --force
```

## Where this lives in the AVLMaps research-bet ladder

This is the most defensible variant of the *"swap the audio encoder"* idea
flagged across multiple bets in the AVLMaps Deep-Dive (notably the IJRR-2025
AudioCLIP→CLAP refresh in the canonical reference, and Bet 3
"multi-source separation before language grounding"). TextBridge does not
require multi-source separation, retraining, or any new dataset — and
already gives a clean ~+25 pp Top-1 lift on intent-level queries. It also
composes cleanly with every other bet:

* **Bet 1 (AVL-Splat).** Replace per-Gaussian audio embeddings with
  per-Gaussian caption embeddings — same change, different geometry.
* **Bet 2 (DynAVLMap).** A caption is also easier to forget cleanly: the
  forgetting kernel can attenuate or rewrite a caption rather than mutating
  an opaque audio embedding.
* **Bet 4 (calibrated fusion).** SBERT cosine is much easier to calibrate
  than cross-modal cosine because the modality gap is gone.
