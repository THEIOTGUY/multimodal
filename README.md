# TextBridge

> A drop-in replacement for the audio-encoder leg of a spatial
> audio-language map. The robot listens, an audio LLM captions what it
> hears in natural language, the caption is re-embedded with a sentence
> encoder, and the map becomes pure **text-text** retrieval — closing the
> modality gap and unlocking instruction-level reasoning that single-vector
> cross-modal cosine cannot express.

---

## 1. What TextBridge is

A household robot wandering an apartment hears sounds: a dog barking, a
faucet running, a baby crying, someone typing. TextBridge gives that robot
two things:

1. A **spatial memory** — every sound source is pinned to a 3D voxel (or a
   2D floor cell), so the robot knows *where* each sound came from.
2. A **language interface** — when the user says *"I want to wash my
   hands"* the robot finds the cell whose remembered sound best matches
   that intent, and walks to it.

The contribution of TextBridge is the **language channel** between sound
and intent. Instead of projecting both audio and text into a shared joint
space and hoping the cosine survives the modality gap, TextBridge converts
the sound into a *natural-language sentence* with an audio LLM, then runs
plain text-text cosine on the result.

---

## 2. Pipeline

```
                  ┌──────────────────────┐
   wav clip ─────►│  Qwen2.5-Omni-7B     │  caption_i
                  │  (audio captioner)   ├──────────┐
                  └──────────────────────┘          │
                                                    │
   user instruction ───┐                            │
                       ▼                            ▼
                  ┌──────────────────────┐  ┌──────────────────────┐
                  │  Sentence-BERT       │  │  Sentence-BERT       │
                  │  all-mpnet-base-v2   │  │  all-mpnet-base-v2   │
                  └──────────────────────┘  └──────────────────────┘
                            │ q_emb (768-d)             │ src_emb (N, 768)
                            └─────────────┬─────────────┘
                                          ▼
                              cos(q_emb, src_emb)         ← text-text cosine
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │ heatmap on the map  │
                                │ (Gaussian blob per  │
                                │  source, weighted)  │
                                └─────────────────────┘
                                          │
                                          ▼
                                       robot goes
                                       to the winner
```

Both sides of the cosine are produced by the **same** encoder (Sentence-
BERT `all-mpnet-base-v2`, 768-d, L2-normalised). The captions are computed
once per source and cached to disk; at query time the robot only runs the
encoder on the user instruction, which is a few-millisecond CPU op.

---

## 3. Components

```
textbridge/
├── captioner.py            — Qwen2.5-Omni-7B → one-sentence caption per wav
├── encoder.py              — Sentence-BERT (all-mpnet-base-v2) wrapper
├── scene.py                — illustrative 5-room floor plan + 9 ESC-50 placements
├── query.py                — heatmap builders, cosine ranking
├── animate.py              — side-by-side robot-navigation MP4 (matplotlib)
├── save_topdown.py         — per-query top-down PNGs (heatmap + path + RGB)
├── render_scene_frames.py  — habitat-sim frame renderer for real Matterport scene
├── generate_placements.py  — placement-only metadata writer (10 scenes × 3 levels)
├── queries_30cat.py        — 300-query bank covering all 30 ESC-50 categories
├── benchmark_real_scene.py — 3,740-query benchmark over the 10 Matterport scenes
├── benchmark_real.json     — full benchmark results (per-scene, per-row)
├── captions_real.json      — Qwen captions cache (180 placed wavs, keyed by path)
├── captions.json           — Qwen captions cache for the illustrative 9 sources
└── textbridge.md           — you are here
```

### 3.1 Captioner (`captioner.py`)

Loads **Qwen2.5-Omni-7B** in float16 with `device_map="auto"` (≈ 21 GB VRAM
checkpoint on first run; cached afterwards). For each wav file, runs the
following prompt:

> *"You are an audio captioning model. Listen to the clip and describe
> what is happening in one rich, descriptive English sentence. Mention the
> likely source object, the activity, and any context that would help a
> household robot reason about who or what is making the sound."*

`disable_talker()` is called to drop the speech head — TextBridge only
needs the text head. Generation uses greedy decoding (`do_sample=False`)
with `max_new_tokens=80`. The first sentence of the output is kept as the
caption and persisted to `captions.json`. After this one-time pass, the
rest of the system is fully GPU-free.

### 3.2 Encoder (`encoder.py`)

A two-line wrapper around `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-mpnet-base-v2")
emb = m.encode(texts, normalize_embeddings=True)   # (N, 768) L2-normalised
```

The L2-normalisation makes the cosine in `query.py` reduce to a plain dot
product. The encoder is loaded once, lru-cached, and reused.

### 3.3 Scene (`scene.py`)

A 12 m × 8 m apartment, voxelised at 0.1 m → an 80 × 120 grid with five
labelled rooms (kitchen, living room, hallway, bedroom, bathroom). Nine
sound sources `S1…S9` are pinned to specific cells, each backed by a real
ESC-50 wav file (so Qwen captions are real, not hand-written). The robot
starts at the centre of the hallway.

### 3.4 Query (`query.py`)

Two functions:

* `rank_sources(query, sources, pipeline)` → returns
  `[(source, cosine), ...]` sorted high-to-low. The `pipeline` argument
  selects what string represents each source on the source side (class
  token vs Qwen caption); the query side is the user's natural-language
  instruction.

* `heatmap(query, sources, pipeline, sigma_cells=5)` → builds a 2D
  similarity heatmap by:
  1. Min-max normalising the cosines into [0, 1].
  2. Cubing them so the winner dominates (`sims ** 3`).
  3. Painting that weight at each source's cell.
  4. Convolving with a Gaussian (σ = 5 cells = 0.5 m).
  5. Re-normalising to [0, 1].

The result is a soft, robot-friendly distribution over the floor plan.

### 3.5 Animator (`animate.py`) and per-query saver (`save_topdown.py`)

`animate.py` produces the headline MP4: for each instruction in the demo
query list, a robot icon walks from the hallway centre to the picked
source over a few seconds (`TRAVEL_S = 1.6`, `HOLD_S = 1.0`, `FPS = 15`),
with the heatmap fading in beneath it. `save_topdown.py` saves the same
information as static PNGs — heatmap + L-shaped Manhattan path + the real
RGB top-down (rooms colour-coded, sources iconised) — one file per query
under `output/topdown/`.

### 3.6 Real-scene bindings (`render_scene_frames.py`, `build_real_map.py`,
`render_real_topdown.py`)

These extend the synthetic apartment to a real photographically-rendered
Matterport scene (`5LpN3gDmAk7_1`) using `habitat-sim`:

* `render_scene_frames.py` walks the scene's pose trajectory in habitat-
  sim with the Matterport `.glb`, saving rgb / depth / semantic frames.
* `build_real_map.py` runs LSeg over the rgb frames and voxelises the
  features into a 3D map (`vlmap/vlmaps.h5df`), then encodes the audio
  clips placed in the scene with AudioCLIP and pins their features to
  voxel positions.
* `render_real_topdown.py` calls `generate_rgb_topdown_map()` to draw the
  scene's photographic top-down, then overlays the TextBridge heatmap for
  each query — the same look as the synthetic demo but with a real RGB
  scene under the heatmap.

---

## 4. Why captions

A class token like `keyboard_typing` lives in a tiny lexical neighbourhood
(*"keyboard"*, *"typing"*) — abstract instructions like *"where can I find
the active home office"* have **no surface overlap** with that token, so
text-text cosine can't find a bridge.

Qwen's caption *"A person typing on a computer keyboard."* shares
*"computer"*, *"person"*, and *"typing"* with the user's intent space.
Sentence-BERT picks that up trivially. Same for *"I want to wash my
hands"* → caption *"A faucet is turned on and off and water is running
into a sink."* shares *"faucet"*, *"water"*, *"sink"* — the cosine just
works.

---

## 5. Evaluation

### 5.1 Methodology

**Dataset.** 10 real Matterport3D scenes (`5LpN3gDmAk7_1`,
`gTV8FGcVJC9_1`, `jh4fc5c5qoQ_1`, `JmbYfDe2QKZ_1`, `JmbYfDe2QKZ_2`,
`mJXqzFtmKg4_1`, `ur6pFq6Qu1A_1`, `UwV83HsGsw3_1`, `Vt2qJdWjCF2_1`,
`YmJkqBEsHnH_1`). Each scene has 20 traversal sub-sequences with
deterministic agent poses. ESC-50 wav clips are placed at agent
positions every ~5 seconds, drawn at random from the active difficulty
level's category pool:

| Level   | Category pool (ESC-50 majors) | # categories |
|---------|--------------------------------|-------------:|
| level_1 | Interior/domestic              | 10           |
| level_2 | level_1 + Human, non-speech    | 20           |
| level_3 | level_2 + Animals              | 30           |

This produces **425 placements** across the 10 scenes × 3 levels, using
**180 unique ESC-50 wav files** (some categories share clips across
scenes / levels). Placements are written into
`<scene>/audio_video/<seq>/range_and_audio_meta_<level>.txt`.

**Three pipelines, identical except for the source-side string fed to
the encoder:**

| Pipeline    | source-side string per placed sound                              |
|-------------|------------------------------------------------------------------|
| `avlmaps`   | ESC-50 class label, underscores → spaces (`"vacuum cleaner"`)    |
| `textbridge`| Qwen2.5-Omni-7B caption of the wav (`"A high-pitched whine…"`)   |
| **`hybrid`**| `"<class>. <caption>"` — class label + caption joined            |

Both sides of every cosine come from the **same** Sentence-BERT encoder
(`sentence-transformers/all-mpnet-base-v2`, 768-d, L2-normalised).
After normalisation, cosine reduces to a dot product:

```
score(query, source) = SBERT(query) · SBERT(source_string)
```

Top-1 prediction is `argmax_source score(query, source)` over the
categories actually placed in that scene at that level. A query is
**correct** iff that argmax equals the ground-truth ESC-50 category.

**Query bank.** 300 abstract / intent-level instructions — exactly 10
per ESC-50 category, covering all 30 level_3 categories. Lives at
`textbridge/queries_30cat.py:QUERIES_30CAT`. Each scene-level pair runs
the queries filtered to its placed categories. Across all 10 scenes ×
3 levels that sums to **3,740 scored queries**.

**Captioner config (used by both `textbridge` and `hybrid`):**

| Field                | Value                                          |
|----------------------|------------------------------------------------|
| Model                | `Qwen/Qwen2.5-Omni-7B`                         |
| Input modality       | raw 16-bit wav (no transcription middle step)  |
| dtype                | `float16` (`device_map="auto"`)                |
| Decoding             | greedy (`do_sample=False`)                     |
| `max_new_tokens`     | 80                                             |
| Speech head          | disabled (`model.disable_talker()`)            |
| Truncation           | first sentence kept                            |
| Cache                | `textbridge/captions_real.json` (180 wavs)     |

The exact captioning prompt is the one shipped in `captioner.py:25`.

### 5.2 Headline numbers

3,740 queries, 10 scenes × 3 levels:

| Pipeline                              | Top-1 Accuracy   | Δ vs `avlmaps` |
|---------------------------------------|-----------------:|---------------:|
| `avlmaps`    (class label only)       | 80.9 %           | —              |
| `textbridge` (Qwen caption only)      | 63.5 %           | **−17.4 pp**   |
| **`hybrid`** (class label + caption)  | **83.6 %**       | **+2.7 pp**    |

`hybrid` is the strongest configuration. **Caption-only retrieval
*regresses* on the real-scene benchmark** — the synthetic apartment in
the demo MP4 over-states the caption advantage because it uses
hand-curated wavs whose Qwen captions happen to bridge cleanly to
intent. On 180 randomly-drawn ESC-50 wavs across 10 scenes the picture
inverts: the class label is more reliable in isolation, *but* the
caption still adds independent signal that the hybrid recovers.

### 5.3 By difficulty level

| Level                    | n   | `avlmaps` | `textbridge` | **`hybrid`** |
|--------------------------|----:|----------:|-------------:|-------------:|
| level_1 (10 cats)        | 920 | 74.8 %    | 51.6 %       | **77.1 %**   |
| level_2 (20 cats)        | 1400| 83.0 %    | 64.9 %       | **85.2 %**   |
| level_3 (30 cats)        | 1420| 82.7 %    | 69.7 %       | **86.3 %**   |

The level_1 pool is the hardest because all 10 categories are *Interior/
domestic* — they share many sound-object words ("door", "click",
"clock", "machine"), so disambiguation depends on having the canonical
class word. Caption-only collapses (−23 pp). Hybrid still wins.

The TextBridge gap narrows as the pool diversifies (−23, −18, −13 pp
for level_1, level_2, level_3) — captions help most when the placed
categories span Interior + Human + Animals, where the class words are
already lexically far apart.

### 5.4 By scene

| Scene              | n   | `avlmaps` | `textbridge` | **`hybrid`** |
|--------------------|----:|----------:|-------------:|-------------:|
| `5LpN3gDmAk7_1`    | 440 | 82.5 %    | 70.0 %       | **84.1 %**   |
| `JmbYfDe2QKZ_1`    | 270 | 84.8 %    | 65.9 %       | 84.8 %       |
| `JmbYfDe2QKZ_2`    | 340 | 80.9 %    | 70.6 %       | **86.2 %**   |
| `UwV83HsGsw3_1`    | 400 | 79.2 %    | 63.0 %       | **82.5 %**   |
| `Vt2qJdWjCF2_1`    | 440 | 78.2 %    | 61.8 %       | **83.0 %**   |
| `YmJkqBEsHnH_1`    | 180 | **88.9 %**| 67.8 %       | 85.6 %       |
| `gTV8FGcVJC9_1`    | 440 | 78.2 %    | 63.2 %       | **82.7 %**   |
| `jh4fc5c5qoQ_1`    | 210 | 84.8 %    | 66.2 %       | **86.7 %**   |
| `mJXqzFtmKg4_1`    | 510 | 79.8 %    | 56.9 %       | **80.4 %**   |
| `ur6pFq6Qu1A_1`    | 510 | 79.8 %    | 57.8 %       | **84.5 %**   |

Hybrid wins or ties in 9 / 10 scenes. The single loss (`YmJkqBEsHnH_1`)
is one of the smallest cells (n = 180) where AVLMaps already saturates
at 88.9 %. The largest hybrid gain over AVLMaps is on `ur6pFq6Qu1A_1`
(+4.7 pp) — a scene where many random placements are
`door_wood_creaks`, `clock_tick`, and `washing_machine`, all of which
have rich Qwen descriptors that complement their class word.

### 5.5 Per-category win/loss profile

How often `textbridge` (caption-only) wins or loses **vs `avlmaps`**:

**Top categories where caption-only LOSES** (caption doesn't say the
class word):

| Category            | losses |
|---------------------|-------:|
| `vacuum_cleaner`    | 105    |
| `door_wood_knock`   | 101    |
| `clock_alarm`       | 96     |
| `can_opening`       | 76     |
| `glass_breaking`    | 76     |
| `washing_machine`   | 75     |
| `breathing`         | 74     |
| `mouse_click`       | 50     |

**Top categories where caption-only WINS** (caption is richer than the
two-word class label):

| Category            | wins |
|---------------------|-----:|
| `door_wood_creaks`  | 61   |
| `clock_tick`        | 52   |
| `washing_machine`   | 51   |
| `vacuum_cleaner`    | 35   |
| `hen`               | 13   |
| `brushing_teeth`    | 11   |
| `keyboard_typing`   | 11   |
| `sheep`             | 11   |

Note that `vacuum_cleaner` and `washing_machine` appear in **both**
columns. That is the diagnostic signal: for direct queries
(*"where is the vacuum cleaner"*) the class label dominates; for intent
queries (*"find the small whining motor"*) the caption dominates.
Hybrid keeps both.

### 5.6 Why hybrid works

Caption-only's failure mode is concentrated:

1. **Class word missing from the caption.** Qwen described several
   `vacuum_cleaner` clips as *"a high-pitched whine emitted by a small
   motor"* — never says "vacuum". Direct queries containing "vacuum"
   then prefer any source whose caption *does* mention a household
   word, often `washing_machine` or `brushing_teeth`.
2. **Two ESC-50 categories share a noun.** `door_wood_knock` and
   `door_wood_creaks` both end up with captions about *"a wooden
   door"*, so the caption-only pipeline can't tell them apart — the
   class label is the only disambiguator.
3. **Caption picks adjacency over identity.** *"the laundry is going"*
   pulls toward `pouring_water`'s "*water is running*" instead of
   `washing_machine`'s "*vibrating sound*".

By concatenating `<class>. <caption>` the source string keeps
the canonical class word for direct queries **and** keeps the rich
descriptors for intent queries. The two failure modes are
near-disjoint, so they cancel — hybrid beats both pipelines on every
difficulty level.

**`hybrid` vs `avlmaps` per-query matrix (n = 3,740):**

| Outcome                | Count |
|------------------------|------:|
| Both correct           | 2,800 |
| Only `hybrid` correct  | **328** |
| Only `avlmaps` correct | 224   |
| Both wrong             | 388   |

Net hybrid gain: **+104 queries (+2.78 pp)**. The 388 "both wrong"
queries are the genuinely hard ones — the next lift would come from
caption-prompt tightening (so Qwen always names the source object) and
from sampling-based caption averaging.

### 5.7 Files produced by the benchmark

```
textbridge/queries_30cat.py        # 300 queries × 30 categories (the bank)
textbridge/captions_real.json      # 180 cached Qwen captions, keyed by wav path
textbridge/benchmark_real.json     # full per-scene-level results + per-row
output/benchmark_real.json         # mirror copy
```

JSON schema highlights:

```json
{
  "n_scenes": 30,
  "n_queries_total": 3740,
  "avlmaps_top1":    0.8086,
  "textbridge_top1": 0.6348,
  "hybrid_top1":     0.8358,
  "per_scene": [
    {
      "scene": "5LpN3gDmAk7_1", "level": "level_3",
      "n_categories": 17, "n_queries": 170,
      "avlmaps_top1": 0.876, "textbridge_top1": 0.853, "hybrid_top1": 0.882,
      "category_caption": { "vacuum_cleaner": "A high-pitched whine …", … },
      "rows": [
        { "query": "find the small whining motor",
          "gt_category": "vacuum_cleaner",
          "avlmaps_pick": "washing_machine", "ok_avlmaps": false,
          "textbridge_pick": "vacuum_cleaner", "ok_textbridge": true,
          "hybrid_pick": "vacuum_cleaner",     "ok_hybrid": true,
          "textbridge_caption_used": "A high-pitched whine …" }
      ]
    }
  ]
}
```

---

## 6. Reproduce

```bash
cd AVLMaps

# 1. Generate placement metadata for all 10 scenes (uses only poses.txt;
#    no rgb rendering / ffmpeg / Matterport mesh extraction required).
PYTHONPATH=. conda run -n avlmaps python -m textbridge.generate_placements
# -> 600 meta files / 425 placements across <scene>/audio_video/<seq>/

# 2. Run the 3,740-query benchmark over all 10 scenes × 3 levels.
#    First pass captions every unique placed wav with Qwen2.5-Omni-7B
#    (~3-5 min on GPU, cached to textbridge/captions_real.json).
#    Subsequent runs read the cache and only redo the SBERT scoring.
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 \
  conda run -n avlmaps python -m textbridge.benchmark_real_scene
# -> textbridge/benchmark_real.json   (per-scene-level + per-row)
# -> output/benchmark_real.json

# Optional: re-caption everything from scratch
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 \
  conda run -n avlmaps python -m textbridge.captioner --force
```

The full benchmark from a clean state takes about 10–15 min (most of it
is the one-time Qwen captioning of 180 wavs). After the cache is built,
re-scoring the 3,740 queries is sub-minute.
