"""Export Markdown, Excel, and LaTeX tables from completed formal runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.paper_artifacts.summarize_ablation import summarize


def generate(root: str | Path, output_dir: str | Path | None = None) -> Path:
    output = Path(output_dir) if output_dir else Path(root) / "paper_summary"
    frame = summarize(root, output)
    (output / "formal_ablation_results.tex").write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.5f}", escape=True),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(generate(args.root, args.output_dir))


if __name__ == "__main__":
    main()

