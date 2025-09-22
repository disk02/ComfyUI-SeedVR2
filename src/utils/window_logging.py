"""Window attention logging helpers for SeedVR2 instrumentation."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple, Literal, Union

Variant = Literal["regular", "shifted"]


@dataclass
class WindowMeta:
    attn_mode: str
    dt: int
    dh: int
    dw: int
    pt: int
    ph: int
    pw: int
    nt: int
    nh: int
    nw: int
    variant: Variant


@dataclass
class WindowPlanDump:
    latent: Tuple[int, int, int]
    window: Tuple[int, int, int]
    counts: Tuple[int, int, int]
    slices_regular: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]
    slices_shifted: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]


@dataclass
class WindowLoggingConfig:
    log_window_info: bool = False
    dump_window_plan: Optional[str] = None
    make_window_overlay: bool = False


@dataclass
class _WindowSummary:
    pt: int
    ph: int
    pw: int
    nt: int
    nh: int
    nw: int


def _slice_bounds(slc: slice) -> Tuple[int, int]:
    start = int(slc.start) if slc.start is not None else 0
    stop = int(slc.stop) if slc.stop is not None else start
    return start, stop


def _summarize_slices(slices: Iterable[Tuple[slice, slice, slice]]) -> _WindowSummary:
    t_pairs = []
    h_pairs = []
    w_pairs = []
    for st, sh, sw in slices:
        t_pairs.append(_slice_bounds(st))
        h_pairs.append(_slice_bounds(sh))
        w_pairs.append(_slice_bounds(sw))

    def _counts_and_span(pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not pairs:
            return 0, 0
        unique_pairs = []
        seen = set()
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        span = 0
        for start, stop in unique_pairs:
            span = max(span, stop - start)
        return len(unique_pairs), span

    nt, pt = _counts_and_span(t_pairs)
    nh, ph = _counts_and_span(h_pairs)
    nw, pw = _counts_and_span(w_pairs)
    return _WindowSummary(pt=pt, ph=ph, pw=pw, nt=nt, nh=nh, nw=nw)


def _to_slice_pairs(slices: Iterable[Tuple[slice, slice, slice]]) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    result: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
    for st, sh, sw in slices:
        result.append((_slice_bounds(st), _slice_bounds(sh), _slice_bounds(sw)))
    return result


def plan_from_slices(
    latent: Tuple[int, int, int],
    regular_slices: Iterable[Tuple[slice, slice, slice]],
    shifted_slices: Iterable[Tuple[slice, slice, slice]],
    summary: Optional[_WindowSummary] = None,
) -> WindowPlanDump:
    if summary is None:
        summary = _summarize_slices(regular_slices)
    reg_pairs = _to_slice_pairs(regular_slices)
    shf_pairs = _to_slice_pairs(shifted_slices)
    return WindowPlanDump(
        latent=latent,
        window=(summary.pt, summary.ph, summary.pw),
        counts=(summary.nt, summary.nh, summary.nw),
        slices_regular=reg_pairs,
        slices_shifted=shf_pairs,
    )


def build_window_metas(
    attn_mode: str,
    latent: Tuple[int, int, int],
    regular_slices: Iterable[Tuple[slice, slice, slice]],
    shifted_slices: Iterable[Tuple[slice, slice, slice]],
) -> Tuple[List[WindowMeta], WindowPlanDump]:
    dt, dh, dw = latent
    regular_list = list(regular_slices)
    shifted_list = list(shifted_slices)
    regular_summary = _summarize_slices(regular_list)
    shifted_summary = _summarize_slices(shifted_list)
    metas = [
        WindowMeta(
            attn_mode=attn_mode,
            dt=dt,
            dh=dh,
            dw=dw,
            pt=regular_summary.pt,
            ph=regular_summary.ph,
            pw=regular_summary.pw,
            nt=regular_summary.nt,
            nh=regular_summary.nh,
            nw=regular_summary.nw,
            variant="regular",
        ),
        WindowMeta(
            attn_mode=attn_mode,
            dt=dt,
            dh=dh,
            dw=dw,
            pt=shifted_summary.pt,
            ph=shifted_summary.ph,
            pw=shifted_summary.pw,
            nt=shifted_summary.nt,
            nh=shifted_summary.nh,
            nw=shifted_summary.nw,
            variant="shifted",
        ),
    ]
    plan = plan_from_slices(latent, regular_list, shifted_list, summary=regular_summary)
    return metas, plan


def maybe_dump_plan(path: Optional[str], plan: WindowPlanDump) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(plan), handle, indent=2)


class WindowLogger:
    """Stateful helper coordinating window attention logging."""

    def __init__(
        self,
        config: Optional[WindowLoggingConfig] = None,
    ) -> None:
        self.config = config or WindowLoggingConfig()
        self._latest_plan: Optional[WindowPlanDump] = None
        self._plan_written: bool = False
        self._overlay_warned: bool = False

    @property
    def enabled(self) -> bool:
        cfg = self.config
        return bool(
            (cfg.dump_window_plan and cfg.dump_window_plan.strip())
            or cfg.log_window_info
            or cfg.make_window_overlay
        )

    def update_config(self, config: Optional[WindowLoggingConfig]) -> None:
        config = config or WindowLoggingConfig()
        path_changed = self.config.dump_window_plan != config.dump_window_plan
        self.config = config
        if path_changed:
            self._plan_written = False
        if not self.enabled:
            self._latest_plan = None

    def begin_capture(self) -> None:
        if not self.enabled:
            self._latest_plan = None
            self._plan_written = False
            return
        self._latest_plan = None
        self._plan_written = False
        self._overlay_warned = False

    def log_meta(self, meta: WindowMeta) -> None:
        if not self.config.log_window_info:
            return
        line = (
            f"[ATTN] ATTN_MODE={meta.attn_mode} latent=[{meta.dt},{meta.dh},{meta.dw}] "
            f"window=[{meta.pt},{meta.ph},{meta.pw}] counts=[{meta.nt},{meta.nh},{meta.nw}] "
            f"variant={meta.variant}"
        )
        print(line)

    def record_plan(self, plan: WindowPlanDump) -> None:
        if not self.enabled:
            return
        self._latest_plan = plan
        if self.config.dump_window_plan and not self._plan_written:
            maybe_dump_plan(self.config.dump_window_plan, plan)
            self._plan_written = True

    def process(
        self,
        attn_mode: str,
        latent: Tuple[int, int, int],
        regular_slices: Iterable[Tuple[slice, slice, slice]],
        shifted_slices: Iterable[Tuple[slice, slice, slice]],
    ) -> None:
        if not self.enabled:
            return
        regular_list = list(regular_slices)
        shifted_list = list(shifted_slices)
        metas, plan = build_window_metas(attn_mode, latent, regular_list, shifted_list)
        for meta in metas:
            self.log_meta(meta)
        self.record_plan(plan)

    def maybe_make_overlay(self, image_path: str, out_path: Optional[str] = None) -> Optional[str]:
        if not self.config.make_window_overlay:
            return None
        if self._latest_plan is None:
            return None
        return draw_window_overlay(image_path, self._latest_plan, out_path=out_path)

    @property
    def latest_plan(self) -> Optional[WindowPlanDump]:
        return self._latest_plan


def attach_window_logger(model, window_logger: Optional[WindowLogger]) -> None:
    if model is None or window_logger is None:
        return
    for module in model.modules():
        if module.__class__.__name__ == "NaSwinAttention":
            module.window_logger = window_logger


def draw_window_overlay(
    image_path: str,
    plan: Union[WindowPlanDump, dict],
    out_path: Optional[str] = None,
    regular_color: Tuple[int, int, int] = (255, 0, 0),
    shifted_color: Tuple[int, int, int] = (0, 255, 0),
    dash_length: int = 6,
) -> str:
    from PIL import Image, ImageDraw

    if isinstance(plan, WindowPlanDump):
        plan_dict = asdict(plan)
    else:
        plan_dict = plan

    _, latent_h, latent_w = plan_dict.get("latent", (1, 1, 1))
    regular_slices = plan_dict.get("slices_regular", [])
    shifted_slices = plan_dict.get("slices_shifted", [])

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def _scale_w(coord: int) -> int:
        if latent_w == 0:
            return 0
        return int(round(coord * width / latent_w))

    def _scale_h(coord: int) -> int:
        if latent_h == 0:
            return 0
        return int(round(coord * height / latent_h))

    def _unique_bounds(pairs: List[Tuple[int, int]]) -> List[int]:
        coords = set()
        for start, stop in pairs:
            coords.add(start)
            coords.add(stop)
        ordered = sorted(coords)
        return ordered

    def _collect_pairs(slice_triplets, index: int) -> List[Tuple[int, int]]:
        return [tuple(triplet[index]) for triplet in slice_triplets if len(triplet) > index]

    regular_w_pairs = _collect_pairs(regular_slices, 2)
    regular_h_pairs = _collect_pairs(regular_slices, 1)
    shifted_w_pairs = _collect_pairs(shifted_slices, 2)
    shifted_h_pairs = _collect_pairs(shifted_slices, 1)

    # Draw regular grid (solid lines)
    for coord in _unique_bounds(regular_w_pairs):
        x = _scale_w(coord)
        draw.line([(x, 0), (x, height)], fill=regular_color, width=1)
    for coord in _unique_bounds(regular_h_pairs):
        y = _scale_h(coord)
        draw.line([(0, y), (width, y)], fill=regular_color, width=1)

    # Draw shifted grid (dashed lines)
    def _draw_dashed_vertical(x: int) -> None:
        y = 0
        while y < height:
            y_end = min(y + dash_length, height)
            draw.line([(x, y), (x, y_end)], fill=shifted_color, width=1)
            y += dash_length * 2

    def _draw_dashed_horizontal(y: int) -> None:
        x = 0
        while x < width:
            x_end = min(x + dash_length, width)
            draw.line([(x, y), (x_end, y)], fill=shifted_color, width=1)
            x += dash_length * 2

    for coord in _unique_bounds(shifted_w_pairs):
        _draw_dashed_vertical(_scale_w(coord))
    for coord in _unique_bounds(shifted_h_pairs):
        _draw_dashed_horizontal(_scale_h(coord))

    out_path = out_path or f"{os.path.splitext(image_path)[0]}_windows.png"
    directory = os.path.dirname(out_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    image.save(out_path)
    return out_path


def load_window_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
