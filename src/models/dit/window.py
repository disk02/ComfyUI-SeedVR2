from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math


# -------------------------
# Data structures
# -------------------------


@dataclass
class WindowSlices:
    """Grouped regular/shifted slice triplets for window partitioning."""

    regular: List[Tuple[slice, slice, slice]]
    shifted: List[Tuple[slice, slice, slice]]


@dataclass
class WindowPlan:
    """Immutable description of an attention window lattice."""

    dt: int
    dh: int
    dw: int
    ph: int
    pw: int
    pt: int
    nh: int
    nw: int
    nt: int
    proxy_h: int
    proxy_w: int
    slices: WindowSlices


# -------------------------
# Helpers (internal)
# -------------------------


def _compute_proxy(dh: int, dw: int, train_token_area: int) -> Tuple[int, int, float]:
    # scale factor from actual (dh,dw) to training proxy area A = 45*80 (SeedVR2 style)
    s = math.sqrt(max(1e-9, train_token_area) / max(1, dh * dw))
    ph = max(1, int(round(dh * s)))
    pw = max(1, int(round(dw * s)))
    return ph, pw, s


def _choose_counts(L: int, target_size: int) -> int:
    """
    Choose number of windows n = ceil(L / target_size) but avoid tiny last stripe.
    We ensure floor(L/n) >= ceil(target_size/2) when n>1 by backing off n if needed.
    """

    n = max(1, int(math.ceil(L / max(1, target_size))))
    half = int(math.ceil(target_size / 2))
    while n > 1 and (L // n) < half:
        n -= 1
    return n


def _balanced_splits(L: int, n: int) -> List[int]:
    """
    Split L into n positive integers that sum to L, sizes differ by at most 1.
    e.g., L=10,n=3 -> [4,3,3]. Deterministic.
    """

    base = L // n
    rem = L - base * n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    # Guarantee all positive
    assert all(sz > 0 for sz in sizes), (L, n, sizes)
    return sizes


def _slices_1d_from_sizes(L: int, sizes: List[int]) -> List[Tuple[int, int]]:
    pts = []
    cur = 0
    for sz in sizes:
        nxt = min(L, cur + sz)
        pts.append((cur, nxt))
        cur = nxt
    # Numeric stability
    if pts:
        pts[-1] = (pts[-1][0], L)
    return pts


def _grid_regular(L: int, target_size: int) -> Tuple[List[Tuple[int, int]], int, int]:
    """Regular (non-shifted) grid: balanced n windows."""

    n = _choose_counts(L, target_size)
    sizes = _balanced_splits(L, n)
    segs = _slices_1d_from_sizes(L, sizes)
    p = int(round(L / n)) if n > 0 else L  # representative window size
    return segs, n, p


def _grid_shifted(L: int, rep_size: int) -> List[Tuple[int, int]]:
    """Return a non-overlapping shifted partition of ``[0, L)``."""

    if L <= rep_size:
        return [(0, L)]

    start = rep_size // 2
    segs: List[Tuple[int, int]] = []

    if start > 0:
        segs.append((0, start))

    cur = start
    while cur + rep_size <= L:
        segs.append((cur, cur + rep_size))
        cur += rep_size

    if cur < L:
        tail_len = L - cur
        half = int(math.ceil(rep_size / 2))
        if tail_len < half and segs:
            head, _ = segs[-1]
            segs[-1] = (head, L)
        else:
            segs.append((cur, L))

    return segs


def _cartesian_3d(
    ts: List[Tuple[int, int]],
    hs: List[Tuple[int, int]],
    ws: List[Tuple[int, int]],
) -> List[Tuple[slice, slice, slice]]:
    out: List[Tuple[slice, slice, slice]] = []
    for t0, t1 in ts:
        for h0, h1 in hs:
            for w0, w1 in ws:
                out.append((slice(t0, t1), slice(h0, h1), slice(w0, w1)))
    return out


def _count_tokens_3d(slices3d: List[Tuple[slice, slice, slice]]) -> int:
    return sum(
        (t.stop - t.start) * (h.stop - h.start) * (w.stop - w.start)
        for (t, h, w) in slices3d
    )


# -------------------------
# Public API
# -------------------------


def compute_adaptive_windows(
    d_t: int,
    d_h: int,
    d_w: int,
    *,
    train_token_area: int = 45 * 80,
    cap_t: int = 30,
    target_nh: int = 3,
    target_nw: int = 3,
) -> WindowPlan:
    """Return a deterministic window tiling using SeedVR2's proxy equations.

    The helper is side-effect free and recomputes the attention lattice from the
    latent dimensions. It mimics the adaptive sizing described for SeedVR2 by
    (1) computing a proxy resolution with area ``train_token_area``, (2) choosing
    counts that avoid extremely small border stripes, and (3) generating both the
    regular and half-window-shifted partitions.
    """

    assert d_t >= 1 and d_h >= 1 and d_w >= 1, (d_t, d_h, d_w)

    # Time axis
    pt_target = min(cap_t, d_t)
    nt = _choose_counts(d_t, pt_target)
    t_sizes = _balanced_splits(d_t, nt)
    t_regular = _slices_1d_from_sizes(d_t, t_sizes)
    pt = int(round(d_t / nt)) if nt > 0 else d_t
    t_shifted = _grid_shifted(d_t, pt)

    # Spatial axes via proxy
    proxy_h, proxy_w, _ = _compute_proxy(d_h, d_w, train_token_area)

    # Target sizes from proxy/targets
    ph_target = max(1, int(math.ceil(proxy_h / max(1, target_nh))))
    pw_target = max(1, int(math.ceil(proxy_w / max(1, target_nw))))

    # Regular grids (balanced)
    h_regular, nh, ph_rep = _grid_regular(d_h, ph_target)
    w_regular, nw, pw_rep = _grid_regular(d_w, pw_target)

    # Shifted grids (uniform step = representative size from regular)
    h_shifted = _grid_shifted(d_h, ph_rep)
    w_shifted = _grid_shifted(d_w, pw_rep)

    # Compose 3D
    regular_3d = _cartesian_3d(t_regular, h_regular, w_regular)
    shifted_3d = _cartesian_3d(t_regular, h_shifted, w_shifted)

    expected_tokens = d_t * d_h * d_w
    regular_tokens = _count_tokens_3d(regular_3d)
    shifted_tokens = _count_tokens_3d(shifted_3d)
    assert (
        regular_tokens == expected_tokens
    ), (regular_tokens, expected_tokens, "regular coverage mismatch")
    assert (
        shifted_tokens == expected_tokens
    ), (shifted_tokens, expected_tokens, "shifted coverage mismatch")

    return WindowPlan(
        dt=d_t,
        dh=d_h,
        dw=d_w,
        pt=pt,
        ph=ph_rep,
        pw=pw_rep,
        nt=nt,
        nh=nh,
        nw=nw,
        proxy_h=proxy_h,
        proxy_w=proxy_w,
        slices=WindowSlices(regular=regular_3d, shifted=shifted_3d),
    )


# ---------------------------------------------------------------------------
# Legacy helpers (retained for compatibility, no behavior change).
# ---------------------------------------------------------------------------


from math import ceil


def get_window_op(name: str):
    if name == "720pwin_by_size_bysize":
        return make_720Pwindows_bysize
    if name == "720pswin_by_size_bysize":
        return make_shifted_720Pwindows_bysize
    raise ValueError(f"Unknown windowing method: {name}")


# -------------------------------- Windowing -------------------------------- #


def make_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    # cal windows under 720p
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, 30) / resized_nt)  # window size.
    nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)  # window size.
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]


def make_shifted_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    # cal windows under 720p
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, 30) / resized_nt)  # window size.

    st, sh, sw = (  # shift size.
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)  # window size.
    nt, nh, nw = (  # number of window.
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]

