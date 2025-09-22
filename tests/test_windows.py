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


def _count_tokens_3d(slices3d):
    return sum(
        (t.stop - t.start) * (h.stop - h.start) * (w.stop - w.start)
        for (t, h, w) in slices3d
    )


def _assert_no_tiny_borders(plan):
    h_segments = _axis_from_3d(plan.slices.regular, "H")
    w_segments = _axis_from_3d(plan.slices.regular, "W")
    min_h = _min_border_segment(h_segments, plan.dh)
    min_w = _min_border_segment(w_segments, plan.dw)
    assert min_h >= math.ceil(plan.ph / 2), f"H border {min_h} < ceil(ph/2)={math.ceil(plan.ph / 2)}"
    assert min_w >= math.ceil(plan.pw / 2), f"W border {min_w} < ceil(pw/2)={math.ceil(plan.pw / 2)}"


def _assert_shifted_covers(plan):
    expected = plan.dt * plan.dh * plan.dw
    assert _count_tokens_3d(plan.slices.regular) == expected
    assert _count_tokens_3d(plan.slices.shifted) == expected


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


def test_shifted_is_partition_1920():
    plan = compute_adaptive_windows(1, 120, 120)
    expected = plan.dt * plan.dh * plan.dw
    assert _count_tokens_3d(plan.slices.regular) == expected
    assert _count_tokens_3d(plan.slices.shifted) == expected


@pytest.mark.parametrize("dh,dw", [(64, 64), (96, 120), (120, 96), (128, 128), (117, 131)])
def test_shifted_is_partition_various(dh, dw):
    plan = compute_adaptive_windows(1, dh, dw)
    expected = plan.dt * plan.dh * plan.dw
    assert _count_tokens_3d(plan.slices.regular) == expected
    assert _count_tokens_3d(plan.slices.shifted) == expected


def test_shifted_axis_contiguity():
    plan = compute_adaptive_windows(1, 120, 96)
    h_segments = sorted({(h.start, h.stop) for (_, h, _) in plan.slices.shifted})
    assert h_segments[0][0] == 0
    assert h_segments[-1][1] == plan.dh
    for (a0, a1), (b0, b1) in zip(h_segments, h_segments[1:]):
        assert a1 == b0

