"""End-to-end demo runner: render the side-by-side MP4 + print eval table.

Assumes captions are already cached at textbridge/captions.json. If not,
run `python -m textbridge.captioner` first (requires GPU + Qwen2.5-Omni-7B).
"""
from __future__ import annotations

from pathlib import Path

from textbridge.animate import render
from textbridge.eval import evaluate, format_report


def main() -> None:
    out = render("output/textbridge_demo.mp4")
    print(f"\n[demo] wrote MP4 -> {out.resolve()}\n")
    res = evaluate()
    report = format_report(res)
    print(report)
    Path("output/textbridge_eval.txt").write_text(report + "\n")
    print(f"\n[demo] wrote report -> output/textbridge_eval.txt")


if __name__ == "__main__":
    main()
