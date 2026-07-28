"""Build same-image, same-scale comparison panels from saved visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def panel(images: list[Path], labels: list[str], output: Path, size: tuple[int, int] = (640, 640)) -> Path:
    loaded = [Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR) for path in images]
    canvas = Image.new("RGB", (size[0] * len(loaded), size[1] + 40), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(loaded, labels, strict=True)):
        canvas.paste(image, (index * size[0], 40))
        draw.text((index * size[0] + 10, 10), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.images) != len(args.labels):
        raise ValueError("--images and --labels must have equal length")
    print(panel([Path(item) for item in args.images], args.labels, Path(args.output)))


if __name__ == "__main__":
    main()
