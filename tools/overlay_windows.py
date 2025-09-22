#!/usr/bin/env python3
"""Render window lattice overlays from SeedVR2 window plans."""
from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.window_logging import draw_window_overlay, load_window_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw SeedVR2 attention window overlays.")
    parser.add_argument("--image", required=True, help="Path to the image to overlay.")
    parser.add_argument("--plan", required=True, help="JSON plan produced by --dump_window_plan.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path (default: <image>_windows.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_window_plan(args.plan)
    out_path = draw_window_overlay(args.image, plan, out_path=args.out)
    print(f"[overlay] wrote {out_path}")


if __name__ == "__main__":
    main()
