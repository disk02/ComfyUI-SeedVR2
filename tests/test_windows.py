import math
import pytest

from src.models.dit.window import compute_adaptive_windows


def _min_border_segment(axis_slices, length):
    """Return the minimum border thickness at head/tail for [0, length)."""

    bounds = sorted({0, length} | {point for start, stop in axis_slices for point in (start, stop)})
    if len(bounds) < 2:
        return length
    head = bounds[1] - bounds[0]
    tail = bounds[-1] - bounds[-2]
    return min(head, tail)


def _axis_from_3d(slices3d, axis: str):
    """Extract sorted unique (start, stop) tuples for a spatial axis."""

    if axis == "H":
        return sorted({(h.start, h.stop) for (_, h, _) in slices3d})
    if axis == "W":
        return sorted({(w.start, w.stop) for (_, _, w) in slices3d})
    raise ValueError(axis)


def _assert_no_tiny_borders(plan):
    h_segments = _axis_from_3d(plan.slices.regular, "H")
    w_segments = _axis_from_3d(plan.slices.regular, "W")
    min_h = _min_border_segment(h_segments, plan.dh)
    min_w = _min_border_segment(w_segments, plan.dw)
    assert min_h >= math.ceil(plan.ph / 2), f"H border {min_h} < ceil(ph/2)={math.ceil(plan.ph / 2)}"
    assert min_w >= math.ceil(plan.pw / 2), f"W border {min_w} < ceil(pw/2)={math.ceil(plan.pw / 2)}"


def _assert_shifted_covers(plan):
    h_regular = _axis_from_3d(plan.slices.regular, "H")
    w_regular = _axis_from_3d(plan.slices.regular, "W")
    h_shifted = _axis_from_3d(plan.slices.shifted, "H")
    w_shifted = _axis_from_3d(plan.slices.shifted, "W")
    assert len(h_shifted) >= len(h_regular), f"shifted nh {len(h_shifted)} < regular nh {len(h_regular)}"
    assert len(w_shifted) >= len(w_regular), f"shifted nw {len(w_shifted)} < regular nw {len(w_regular)}"


@pytest.mark.parametrize("out", [1024, 1536, 1920, 2048])
def test_square_latents(out):
    latent = out // 16
    plan = compute_adaptive_windows(1, latent, latent)
    _assert_no_tiny_borders(plan)
    _assert_shifted_covers(plan)


@pytest.mark.parametrize("width,height", [(2560, 1440), (3840, 1600), (1920, 1080)])
def test_rectangular_latents(width, height):
    dh, dw = height // 16, width // 16
    plan = compute_adaptive_windows(1, dh, dw)
    _assert_no_tiny_borders(plan)
    _assert_shifted_covers(plan)


@pytest.mark.parametrize("frames", [1, 2, 4])
@pytest.mark.parametrize("height", [1080, 1440])
def test_multiframe_latents(frames, height):
    latent = height // 16
    plan = compute_adaptive_windows(frames, latent, latent)
    assert plan.nt >= 1
    assert plan.pt >= 1
    _assert_no_tiny_borders(plan)
    _assert_shifted_covers(plan)


def test_smoke_message():
    plan = compute_adaptive_windows(1, 120, 120)
    msg = (
        f"(dt,dh,dw)=({plan.dt},{plan.dh},{plan.dw}) -> "
        f"(proxy_h,proxy_w)=({plan.proxy_h},{plan.proxy_w}) -> "
        f"(pt,ph,pw)=({plan.pt},{plan.ph},{plan.pw}) x "
        f"(nt,nh,nw)=({plan.nt},{plan.nh},{plan.nw})"
    )
    assert isinstance(msg, str) and msg

