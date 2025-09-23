"""Window helpers and resolver for DiT v2 configs."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from src.models.dit.window import (
    WindowPlan,
    WindowSlices,
    compute_adaptive_windows,
    make_720Pwindows_bysize as _make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize as _make_shifted_720Pwindows_bysize,
)

Slice3D = Tuple[slice, slice, slice]
WindowBuilder = Callable[[Tuple[int, int, int], Tuple[int, int, int]], List[Slice3D]]


def _normalize_counts(num_windows: Tuple[int, int, int]) -> Tuple[int, int, int]:
    nt, nh, nw = num_windows
    return max(1, int(nt)), max(1, int(nh)), max(1, int(nw))


def win_by_size_bysize(
    size: Tuple[int, int, int], num_windows: Tuple[int, int, int]
) -> List[Slice3D]:
    """Regular non-shifted window grid with ceil-sized strides."""

    t, h, w = size
    nt, nh, nw = _normalize_counts(num_windows)
    wt = max(1, (t + nt - 1) // nt)
    wh = max(1, (h + nh - 1) // nh)
    ww = max(1, (w + nw - 1) // nw)

    out: List[Slice3D] = []
    for it in range(nt):
        st = it * wt
        et = min(st + wt, t)
        if et <= st:
            continue
        for ih in range(nh):
            sh = ih * wh
            eh = min(sh + wh, h)
            if eh <= sh:
                continue
            for iw in range(nw):
                sw = iw * ww
                ew = min(sw + ww, w)
                if ew <= sw:
                    continue
                out.append((slice(st, et), slice(sh, eh), slice(sw, ew)))
    return out


def swin_by_size_bysize(
    size: Tuple[int, int, int], num_windows: Tuple[int, int, int]
) -> List[Slice3D]:
    """Half-window shifted lattice following Swin attention conventions."""

    t, h, w = size
    nt, nh, nw = _normalize_counts(num_windows)
    wt = max(1, (t + nt - 1) // nt)
    wh = max(1, (h + nh - 1) // nh)
    ww = max(1, (w + nw - 1) // nw)

    st = 0 if wt >= t else wt // 2
    sh = 0 if wh >= h else wh // 2
    sw = 0 if ww >= w else ww // 2

    out: List[Slice3D] = []
    it = 0
    while True:
        a = it * wt + st
        b = min(a + wt, t)
        if b <= a:
            break
        ih = 0
        while True:
            c = ih * wh + sh
            d = min(c + wh, h)
            if d <= c:
                break
            iw = 0
            while True:
                e = iw * ww + sw
                f = min(e + ww, w)
                if f <= e:
                    break
                out.append((slice(a, b), slice(c, d), slice(e, f)))
                iw += 1
            ih += 1
        it += 1
    return out


# Reuse the project-provided 720p-scaled builders to preserve behavior.
make_720Pwindows_bysize = _make_720Pwindows_bysize
make_shifted_720Pwindows_bysize = _make_shifted_720Pwindows_bysize


_WINDOW_IMPLS: Dict[str, WindowBuilder] = {
    "win_by_size_bysize": win_by_size_bysize,
    "swin_by_size_bysize": swin_by_size_bysize,
    "720pwin_by_size_bysize": make_720Pwindows_bysize,
    "720pswin_by_size_bysize": make_shifted_720Pwindows_bysize,
}


def get_window_op(name: str) -> WindowBuilder:
    """Return the window builder callable associated with ``name``."""

    key = (name or "").strip().lower()
    if not key:
        key = "720pwin_by_size_bysize"
    func = _WINDOW_IMPLS.get(key)
    if func is None:
        valid = ", ".join(sorted(_WINDOW_IMPLS.keys()))
        raise KeyError(f"Unknown window_method='{name}'. Valid: {valid}")
    return func


__all__ = [
    "get_window_op",
    "win_by_size_bysize",
    "swin_by_size_bysize",
    "make_720Pwindows_bysize",
    "make_shifted_720Pwindows_bysize",
    "WindowPlan",
    "WindowSlices",
    "compute_adaptive_windows",
]
