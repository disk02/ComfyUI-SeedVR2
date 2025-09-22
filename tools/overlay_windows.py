#!/usr/bin/env python3
"""Overlay attention window boundaries onto an image."""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw attention window grids over an image.")
    parser.add_argument("--image", required=True, help="Path to the generated image (PNG).")
    parser.add_argument("--lattice", required=True, help="Path to the lattice JSON emitted by the CLI.")
    parser.add_argument("--save", default=None, help="Destination path for the overlay image.")
    parser.add_argument(
        "--latent_downsample",
        type=int,
        default=8,
        help="Spatial downsample factor between latent tokens and pixels (default: 8).",
    )
    return parser.parse_args()


def _regular_boundaries(latent: Sequence[int], window_size: Sequence[int], counts: Sequence[int]) -> Tuple[List[int], List[int]]:
    if len(latent) < 3 or len(window_size) < 3 or len(counts) < 3:
        return [], []
    _, h, w = (int(latent[0]), int(latent[1]), int(latent[2]))
    _, wh, ww = (int(window_size[0]), int(window_size[1]), int(window_size[2]))
    _, nh, nw = (int(counts[0]), int(counts[1]), int(counts[2]))

    col_boundaries = {0, w}
    row_boundaries = {0, h}

    for iw in range(max(nw, 0)):
        start = iw * ww
        end = min((iw + 1) * ww, w)
        col_boundaries.update({start, end})
    for ih in range(max(nh, 0)):
        start = ih * wh
        end = min((ih + 1) * wh, h)
        row_boundaries.update({start, end})
    return sorted(col_boundaries), sorted(row_boundaries)


def _shifted_boundaries(latent: Sequence[int], window_size: Sequence[int], counts: Sequence[int]) -> Tuple[List[int], List[int]]:
    if len(latent) < 3 or len(window_size) < 3 or len(counts) < 3:
        return [], []
    t, h, w = (int(latent[0]), int(latent[1]), int(latent[2]))
    wt, wh, ww = (int(window_size[0]), int(window_size[1]), int(window_size[2]))
    nt, nh, nw = (int(counts[0]), int(counts[1]), int(counts[2]))

    wt = max(wt, 1)
    wh = max(wh, 1)
    ww = max(ww, 1)

    st = 0.5 if wt < t else 0.0
    sh = 0.5 if wh < h else 0.0
    sw = 0.5 if ww < w else 0.0

    col_boundaries = {0, w}
    row_boundaries = {0, h}

    for iw in range(max(nw, 0)):
        start = max(int((iw - sw) * ww), 0)
        end = min(int((iw - sw + 1) * ww), w)
        col_boundaries.update({start, end})
    for ih in range(max(nh, 0)):
        start = max(int((ih - sh) * wh), 0)
        end = min(int((ih - sh + 1) * wh), h)
        row_boundaries.update({start, end})
    return sorted(col_boundaries), sorted(row_boundaries)


def _to_pixels(values: Iterable[int], scale: int, limit: int) -> List[int]:
    pixels = []
    for value in values:
        px = int(round(value * scale))
        px = max(0, min(px, limit))
        pixels.append(px)
    return sorted(set(pixels))


def _draw_dashed_line(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], fill, width: int, dash: int = 12, gap: int = 8) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        sx1 = x1 + dx * distance
        sy1 = y1 + dy * distance
        sx2 = x1 + dx * segment_end
        sy2 = y1 + dy * segment_end
        draw.line((sx1, sy1, sx2, sy2), fill=fill, width=width)
        distance += dash + gap


def _accumulate_boundaries(lattice_entries: Sequence[dict]) -> Tuple[List[int], List[int], List[int], List[int]]:
    regular_cols, regular_rows = set(), set()
    shifted_cols, shifted_rows = set(), set()

    for entry in lattice_entries:
        latent = entry.get("latent")
        regular = entry.get("regular", {})
        shifted = entry.get("shifted", {})

        cols, rows = _regular_boundaries(latent or [], regular.get("p", []), regular.get("n", []))
        regular_cols.update(cols)
        regular_rows.update(rows)

        cols_shift, rows_shift = _shifted_boundaries(latent or [], shifted.get("p", []), shifted.get("n", []))
        shifted_cols.update(cols_shift)
        shifted_rows.update(rows_shift)

    return (
        sorted(regular_cols),
        sorted(regular_rows),
        sorted(shifted_cols),
        sorted(shifted_rows),
    )


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image)
    lattice_path = Path(args.lattice)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not lattice_path.exists():
        raise FileNotFoundError(f"Lattice file not found: {lattice_path}")

    with lattice_path.open("r", encoding="utf-8") as handle:
        lattice_data = json.load(handle)
    if not isinstance(lattice_data, list):
        lattice_data = [lattice_data]

    regular_cols, regular_rows, shifted_cols, shifted_rows = _accumulate_boundaries(lattice_data)

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    scale = max(int(args.latent_downsample), 1)

    reg_cols_px = _to_pixels(regular_cols, scale, width)
    reg_rows_px = _to_pixels(regular_rows, scale, height)
    shift_cols_px = _to_pixels(shifted_cols, scale, width)
    shift_rows_px = _to_pixels(shifted_rows, scale, height)

    regular_color = (255, 0, 0, 160)
    shifted_color = (0, 192, 255, 160)

    for x in reg_cols_px:
        draw.line((x, 0, x, height), fill=regular_color, width=2)
    for y in reg_rows_px:
        draw.line((0, y, width, y), fill=regular_color, width=2)

    for x in shift_cols_px:
        _draw_dashed_line(draw, (x, 0), (x, height), shifted_color, width=2)
    for y in shift_rows_px:
        _draw_dashed_line(draw, (0, y), (width, y), shifted_color, width=2)

    save_path = Path(args.save) if args.save else image_path.with_name(f"{image_path.stem}_overlay.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGBA").save(save_path)
    print(f"Overlay saved to: {save_path}")


if __name__ == "__main__":
    main()
