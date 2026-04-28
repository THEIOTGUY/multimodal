"""100-query Top-1 retrieval benchmark.

Compares two strategies for matching a natural-language instruction to one
of 9 spatially-placed sound sources:
  * `class-token`     — the source string is the ESC-50-style class label
                        (e.g. "dog", "washing machine").
  * `qwen-caption`    — the source string is Qwen2.5-Omni-7B's natural-
                        language caption of the wav file.

Both strategies use the SAME text encoder (Sentence-BERT all-mpnet-base-v2),
so the only thing varying is the source-side string. This isolates the
contribution of the caption bridge.

Usage:
    PYTHONPATH=. conda run -n avlmaps python -m textbridge.benchmark_100
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from textbridge.captioner import CAPTIONS_PATH
from textbridge.query import rank_sources
from textbridge.scene import attach_captions


# 100 abstract / intent-level instructions, ~11 per source. Mix of literal,
# semi-abstract, and intent-only phrasings.
QUERIES_100: List[Tuple[str, str]] = [
    # ---------- S1 dog (class: dog) ----------
    ("where is the dog",                                      "S1"),
    ("where is the pet making noise",                         "S1"),
    ("find the barking animal",                               "S1"),
    ("what creature is being loud",                           "S1"),
    ("I hear an animal in the house",                         "S1"),
    ("the pet wants attention",                               "S1"),
    ("who is making barking sounds",                          "S1"),
    ("locate the source of the bark",                         "S1"),
    ("is there a dog in the house",                           "S1"),
    ("the pup is loud right now",                             "S1"),
    ("find where the woof is coming from",                    "S1"),

    # ---------- S2 vacuum_cleaner ----------
    ("where is the vacuum cleaner",                           "S2"),
    ("find the small whining motor",                          "S2"),
    ("which cleaning device is running",                      "S2"),
    ("I hear a high-pitched motor",                           "S2"),
    ("who is doing the floor cleaning",                       "S2"),
    ("find the appliance with a whining sound",               "S2"),
    ("where is someone vacuuming",                            "S2"),
    ("the floor cleaner is on",                               "S2"),
    ("find the source of the buzz",                           "S2"),
    ("which device sounds like a small engine",               "S2"),
    ("who is vacuuming the carpet",                           "S2"),

    # ---------- S3 glass_breaking ----------
    ("what just shattered",                                   "S3"),
    ("find the broken glass",                                 "S3"),
    ("where was that loud crash",                             "S3"),
    ("something fell and broke",                              "S3"),
    ("which appliance just had a loud accident",              "S3"),
    ("find the source of the clattering",                     "S3"),
    ("who broke something",                                   "S3"),
    ("where is the broken object",                            "S3"),
    ("I heard glass shatter",                                 "S3"),
    ("something just dropped and clanked",                    "S3"),
    ("where is the kitchen disaster",                         "S3"),

    # ---------- S4 washing_machine ----------
    ("find the washing machine",                              "S4"),
    ("which appliance might be malfunctioning",               "S4"),
    ("where is the laundry running",                          "S4"),
    ("find the vibrating appliance",                          "S4"),
    ("who started the wash cycle",                            "S4"),
    ("the laundry is going",                                  "S4"),
    ("find the source of the squeal",                         "S4"),
    ("which device is shaking",                               "S4"),
    ("where is the heavy-duty appliance running",             "S4"),
    ("find the rumbling device",                              "S4"),
    ("I hear something vibrating",                            "S4"),

    # ---------- S5 keyboard_typing ----------
    ("who is typing",                                         "S5"),
    ("where is the keyboard",                                 "S5"),
    ("where can I find the active home office",               "S5"),
    ("find someone using a computer",                         "S5"),
    ("who is at the desk",                                    "S5"),
    ("the home office is occupied",                           "S5"),
    ("find the workspace in use",                             "S5"),
    ("who is doing computer work",                            "S5"),
    ("where is the click-clack of typing",                    "S5"),
    ("someone is at the keyboard",                            "S5"),
    ("find the active workstation",                           "S5"),

    # ---------- S6 crying_baby ----------
    ("the baby is crying",                                    "S6"),
    ("what child needs comfort right now",                    "S6"),
    ("find the infant",                                       "S6"),
    ("who needs to be soothed",                               "S6"),
    ("the kid is upset",                                      "S6"),
    ("where is the baby",                                     "S6"),
    ("who is wailing",                                        "S6"),
    ("find the source of the crying",                         "S6"),
    ("the toddler is fussy",                                  "S6"),
    ("who needs a parent",                                    "S6"),
    ("where is the unhappy little one",                       "S6"),

    # ---------- S7 door_wood_knock ----------
    ("who is at the door",                                    "S7"),
    ("who is at the front entrance",                          "S7"),
    ("someone is knocking",                                   "S7"),
    ("find the visitor at the door",                          "S7"),
    ("I hear knocking",                                       "S7"),
    ("is someone trying to enter",                            "S7"),
    ("who wants to come in",                                  "S7"),
    ("find the source of the rapping sound",                  "S7"),
    ("the front entrance has someone",                        "S7"),
    ("answer the door",                                       "S7"),
    ("who is tapping on the wood",                            "S7"),

    # ---------- S8 brushing_teeth ----------
    ("who is brushing their teeth",                           "S8"),
    ("who is doing their morning hygiene",                    "S8"),
    ("find the bathroom activity",                            "S8"),
    ("someone is in the bathroom",                            "S8"),
    ("find the morning routine in progress",                  "S8"),
    ("who is at the sink for hygiene",                        "S8"),
    ("where is the toothbrush being used",                    "S8"),
    ("find the dental care happening",                        "S8"),
    ("someone is doing oral hygiene",                         "S8"),
    ("who is in the bathroom brushing",                       "S8"),
    ("find where teeth are being cleaned",                    "S8"),

    # ---------- S9 pouring_water (sink) ----------
    ("I want to wash my hands",                               "S9"),
    ("where can I find the sink",                             "S9"),
    ("the faucet is on",                                      "S9"),
    ("I hear water running",                                  "S9"),
    ("where is water flowing",                                "S9"),
    ("find the running tap",                                  "S9"),
    ("I need to rinse off",                                   "S9"),
    ("where can I get clean water",                           "S9"),
    ("find the source of the water sound",                    "S9"),
    ("the kitchen tap is on",                                 "S9"),
    ("I need to use the sink",                                "S9"),
]


def evaluate() -> Dict:
    captions = json.loads(CAPTIONS_PATH.read_text())
    sources = attach_captions(captions)
    sid2src = {s.sid: s for s in sources}

    rows = []
    cls_top1 = cap_top1 = 0
    by_sid_cls = defaultdict(lambda: [0, 0])  # [hits, total]
    by_sid_cap = defaultdict(lambda: [0, 0])
    for q, gt in QUERIES_100:
        cls_pick, cls_cos = rank_sources(q, sources, pipeline="avlmaps")[0]
        cap_pick, cap_cos = rank_sources(q, sources, pipeline="textbridge")[0]
        ok_cls = cls_pick.sid == gt
        ok_cap = cap_pick.sid == gt
        cls_top1 += int(ok_cls)
        cap_top1 += int(ok_cap)
        by_sid_cls[gt][1] += 1; by_sid_cls[gt][0] += int(ok_cls)
        by_sid_cap[gt][1] += 1; by_sid_cap[gt][0] += int(ok_cap)
        rows.append({
            "query": q, "gt": gt, "gt_class": sid2src[gt].class_label,
            "class_token_pick": cls_pick.class_label, "ok_class_token": ok_cls,
            "qwen_caption_pick": cap_pick.class_label, "ok_qwen_caption": ok_cap,
            "class_token_cos": cls_cos, "qwen_caption_cos": cap_cos,
        })
    n = len(rows)
    return {
        "n": n,
        "class_token_top1": cls_top1 / n,
        "qwen_caption_top1": cap_top1 / n,
        "by_sid_class_token": dict(by_sid_cls),
        "by_sid_qwen_caption": dict(by_sid_cap),
        "rows": rows,
        "sources": sources,
    }


def format_report(res: Dict) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(" 100-query Top-1 retrieval benchmark — class-token vs Qwen-caption")
    lines.append("    Encoder (both): Sentence-BERT all-mpnet-base-v2 (768-d), L2-normalised cosine")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  class-token  Top-1 = {res['class_token_top1']*100:5.1f}%  "
                 f"({sum(r['ok_class_token']  for r in res['rows'])} / {res['n']})")
    lines.append(f"  qwen-caption Top-1 = {res['qwen_caption_top1']*100:5.1f}%  "
                 f"({sum(r['ok_qwen_caption'] for r in res['rows'])} / {res['n']})")
    lines.append("")

    lines.append("  Per-source breakdown:")
    lines.append(f"  {'sid':<4} {'class':<18} {'class-token':>14} {'qwen-caption':>15}")
    lines.append("  " + "-" * 56)
    sid2src = {s.sid: s for s in res["sources"]}
    for sid in sorted(res["by_sid_class_token"].keys()):
        h_cls, n_cls = res["by_sid_class_token"][sid]
        h_cap, n_cap = res["by_sid_qwen_caption"][sid]
        lines.append(
            f"  {sid:<4} {sid2src[sid].class_label:<18} "
            f"{h_cls:>5}/{n_cls:<2} ({h_cls/n_cls*100:5.1f}%) "
            f"{h_cap:>5}/{n_cap:<2} ({h_cap/n_cap*100:5.1f}%)"
        )
    lines.append("")

    fail_only_cls = [r for r in res["rows"] if not r["ok_class_token"] and r["ok_qwen_caption"]]
    fail_only_cap = [r for r in res["rows"] if r["ok_class_token"] and not r["ok_qwen_caption"]]
    lines.append(f"  caption wins where class-token loses: {len(fail_only_cls)}")
    lines.append(f"  class-token wins where caption loses: {len(fail_only_cap)}")
    lines.append("")
    lines.append("  Sample queries where ONLY qwen-caption gets the right source:")
    for r in fail_only_cls[:8]:
        lines.append(
            f"    GT={r['gt']:>3} ({r['gt_class']:>14}) "
            f"q='{r['query'][:46]:<46}' "
            f"class-token→{r['class_token_pick']:<16} "
            f"qwen→{r['qwen_caption_pick']}"
        )
    if fail_only_cap:
        lines.append("")
        lines.append("  Sample queries where ONLY class-token gets the right source:")
        for r in fail_only_cap[:8]:
            lines.append(
                f"    GT={r['gt']:>3} ({r['gt_class']:>14}) "
                f"q='{r['query'][:46]:<46}' "
                f"class-token→{r['class_token_pick']:<16} "
                f"qwen→{r['qwen_caption_pick']}"
            )
    lines.append("=" * 100)
    return "\n".join(lines)


def _build_detailed_json(res: Dict, captions: Dict[str, str]) -> Dict:
    sid2src = {s.sid: s for s in res["sources"]}
    fail_only_cap = sorted(
        [r for r in res["rows"] if not r["ok_qwen_caption"] and r["ok_class_token"]],
        key=lambda r: r["gt"],
    )
    fail_only_cls = sorted(
        [r for r in res["rows"] if not r["ok_class_token"] and r["ok_qwen_caption"]],
        key=lambda r: r["gt"],
    )
    return {
        "metadata": {
            "task": "Top-1 retrieval — natural-language instruction → spatial sound source",
            "n_queries": res["n"],
            "n_sources": len(res["sources"]),
            "encoder_both_sides": "sentence-transformers/all-mpnet-base-v2 (768-d, L2-normalised)",
            "audio_captioner": {
                "model": "Qwen/Qwen2.5-Omni-7B",
                "dtype": "float16",
                "decoding": "greedy (do_sample=False)",
                "max_new_tokens": 80,
                "prompt": (
                    "You are an audio captioning model. Listen to the clip and describe what is "
                    "happening in one rich, descriptive English sentence. Mention the likely source "
                    "object, the activity, and any context that would help a household robot reason "
                    "about who or what is making the sound."
                ),
                "captions_cache": str(CAPTIONS_PATH),
            },
            "pipelines": {
                "avlmaps_class_token": (
                    "AVLMaps baseline. Source-side string is the ESC-50 class label "
                    "(e.g. 'vacuum_cleaner' → 'vacuum cleaner'). Embedded with the same "
                    "Sentence-BERT encoder so the only thing varying is the source string."
                ),
                "textbridge_qwen_caption": (
                    "TextBridge. Source-side string is the Qwen2.5-Omni-7B caption of the wav "
                    "clip (e.g. 'A high-pitched whine is being emitted by a small motor.'). "
                    "Same Sentence-BERT encoder."
                ),
            },
        },
        "results": {
            "avlmaps_class_token_top1":   res["class_token_top1"],
            "textbridge_qwen_caption_top1": res["qwen_caption_top1"],
            "delta_top1":                 res["qwen_caption_top1"] - res["class_token_top1"],
            "by_source": {
                sid: {
                    "class_label": sid2src[sid].class_label,
                    "qwen_caption": captions.get(sid),
                    "avlmaps_class_token":   {
                        "hits":  res["by_sid_class_token"][sid][0],
                        "total": res["by_sid_class_token"][sid][1],
                        "acc":   res["by_sid_class_token"][sid][0] / res["by_sid_class_token"][sid][1],
                    },
                    "textbridge_qwen_caption": {
                        "hits":  res["by_sid_qwen_caption"][sid][0],
                        "total": res["by_sid_qwen_caption"][sid][1],
                        "acc":   res["by_sid_qwen_caption"][sid][0] / res["by_sid_qwen_caption"][sid][1],
                    },
                }
                for sid in sorted(res["by_sid_class_token"].keys())
            },
            "n_caption_wins_only":     len(fail_only_cls),
            "n_class_token_wins_only": len(fail_only_cap),
        },
        "rows": [
            {
                "query": r["query"],
                "gt_sid": r["gt"],
                "gt_class": r["gt_class"],
                "avlmaps": {
                    "pick_class": r["class_token_pick"],
                    "ok": r["ok_class_token"],
                    "cosine": r["class_token_cos"],
                },
                "textbridge": {
                    "pick_class": r["qwen_caption_pick"],
                    "ok": r["ok_qwen_caption"],
                    "cosine": r["qwen_caption_cos"],
                },
            }
            for r in res["rows"]
        ],
        "caption_wins_only": [
            {"query": r["query"], "gt": r["gt"], "gt_class": r["gt_class"],
             "avlmaps_pick": r["class_token_pick"], "textbridge_pick": r["qwen_caption_pick"]}
            for r in fail_only_cls
        ],
        "class_token_wins_only": [
            {"query": r["query"], "gt": r["gt"], "gt_class": r["gt_class"],
             "avlmaps_pick": r["class_token_pick"], "textbridge_pick": r["qwen_caption_pick"]}
            for r in fail_only_cap
        ],
    }


def main() -> None:
    captions = json.loads(CAPTIONS_PATH.read_text())
    res = evaluate()
    report = format_report(res)
    print(report)

    txt_out = Path("output/benchmark_100.txt")
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.write_text(report + "\n")
    print(f"\n[benchmark] wrote -> {txt_out.resolve()}")

    detailed = _build_detailed_json(res, captions)
    here = Path(__file__).resolve().parent
    json_out_local = here / "benchmark.json"
    json_out_local.write_text(json.dumps(detailed, indent=2))
    print(f"[benchmark] wrote detailed -> {json_out_local}")

    # also a copy in output/ for batch runs
    json_out_run = Path("output/benchmark_100.json")
    json_out_run.write_text(json.dumps(detailed, indent=2))


if __name__ == "__main__":
    main()
