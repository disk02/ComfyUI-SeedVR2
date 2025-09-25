# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from math import ceil
from typing import Dict, NamedTuple, Optional, Tuple
import math

import torch


class SpatialBlendMaskResult(NamedTuple):
    mask: Optional[torch.Tensor]
    closeness: Optional[torch.Tensor]
    policy: str


_SPATIAL_BLEND_SETTINGS: Dict[str, object] = {
    "policy": "hann",
    "margin": 24,
    "min_value": 1e-3,
    "tukey_alpha": 0.5,
    "softmax_tau": 8.0,
    "post_stitch_norm": "none",
    "post_stitch_gn_groups": 16,
    "debug": False,
}
_WINDOW_HALO_LATENT: int = 16
_MASK_CACHE: Dict[
    Tuple[str, int, int, int, int, int, int, int, float, float],
    torch.Tensor,
] = {}
_SOFTMAX_CACHE: Dict[
    Tuple[int, int, int, int, int, int, int],
    torch.Tensor,
] = {}


def set_spatial_blend(
    policy: str,
    margin: int,
    *,
    min_value: float = 1e-3,
    tukey_alpha: float = 0.5,
    softmax_tau: float = 8.0,
    post_stitch_norm: str = "none",
    post_stitch_gn_groups: int = 16,
    debug_enabled: bool = False,
) -> None:
    if policy not in {"off", "hann", "tukey", "softmax"}:
        raise ValueError(f"Unsupported spatial blend policy: {policy}")

    margin = max(int(margin), 0)
    min_value = float(min_value)
    if min_value < 0.0 or min_value >= 1.0:
        raise ValueError("spatial_blend_min must be in [0, 1).")

    tukey_alpha = float(tukey_alpha)
    if tukey_alpha < 0.0:
        tukey_alpha = 0.0
    if tukey_alpha > 1.0:
        tukey_alpha = 1.0

    softmax_tau = float(softmax_tau)
    if softmax_tau <= 0.0:
        softmax_tau = 1e-6

    if post_stitch_norm not in {"none", "layernorm", "groupnorm"}:
        raise ValueError(f"Unsupported post-stitch normalization: {post_stitch_norm}")

    post_stitch_gn_groups = max(int(post_stitch_gn_groups), 1)

    _SPATIAL_BLEND_SETTINGS.update(
        {
            "policy": policy,
            "margin": margin,
            "min_value": min_value,
            "tukey_alpha": tukey_alpha,
            "softmax_tau": softmax_tau,
            "post_stitch_norm": post_stitch_norm,
            "post_stitch_gn_groups": post_stitch_gn_groups,
            "debug": bool(debug_enabled),
        }
    )


def get_spatial_blend_settings() -> Dict[str, object]:
    return _SPATIAL_BLEND_SETTINGS.copy()


def set_window_halo_latent(halo: int) -> None:
    global _WINDOW_HALO_LATENT
    _WINDOW_HALO_LATENT = max(int(halo), 0)


def get_window_halo_latent() -> int:
    return _WINDOW_HALO_LATENT


def _hann_edge_weights_1d(
    length: int,
    base_margin: int,
    left_margin: int,
    right_margin: int,
    *,
    min_value: float = 1e-3,
) -> torch.Tensor:
    if length <= 0:
        return torch.ones(0, dtype=torch.float32)

    weights = torch.ones(length, dtype=torch.float32)
    if base_margin <= 0:
        return weights

    left = min(max(int(left_margin), 0), length)
    right = min(max(int(right_margin), 0), length)

    if left > 0:
        ramp = torch.linspace(0.0, math.pi, left, dtype=torch.float32)
        ramp = min_value + (1.0 - min_value) * 0.5 * (1.0 - torch.cos(ramp))
        weights[:left] = ramp
    if right > 0:
        ramp = torch.linspace(0.0, math.pi, right, dtype=torch.float32)
        ramp = min_value + (1.0 - min_value) * 0.5 * (1.0 - torch.cos(ramp))
        weights[-right:] = torch.flip(ramp, dims=[0])
    return weights


def _tukey_edge_weights_1d(
    length: int,
    base_margin: int,
    left_margin: int,
    right_margin: int,
    *,
    alpha: float,
    min_value: float = 1e-3,
) -> torch.Tensor:
    if length <= 0:
        return torch.ones(0, dtype=torch.float32)

    weights = torch.ones(length, dtype=torch.float32)
    if base_margin <= 0:
        return weights

    alpha = max(0.0, min(1.0, float(alpha)))

    def _apply_side(margin: int, reverse: bool) -> None:
        if margin <= 0:
            return
        ramp_len = int(round(margin * alpha)) if alpha > 0.0 else 0
        if alpha > 0.0 and ramp_len == 0:
            ramp_len = 1
        ramp_len = min(ramp_len, margin)
        plateau_len = margin - ramp_len

        if not reverse:
            start_idx = 0
            if ramp_len > 0:
                ramp = torch.linspace(0.0, math.pi, ramp_len, dtype=torch.float32)
                vals = min_value + (1.0 - min_value) * 0.5 * (1.0 - torch.cos(ramp))
                weights[start_idx:start_idx + ramp_len] = vals
            if plateau_len > 0:
                weights[start_idx + ramp_len:start_idx + ramp_len + plateau_len] = 1.0
        else:
            start_idx = length - margin
            if plateau_len > 0:
                weights[start_idx:start_idx + plateau_len] = 1.0
            if ramp_len > 0:
                ramp = torch.linspace(0.0, math.pi, ramp_len, dtype=torch.float32)
                vals = min_value + (1.0 - min_value) * 0.5 * (1.0 - torch.cos(ramp))
                weights[length - ramp_len:] = torch.flip(vals, dims=[0])

    _apply_side(min(max(int(left_margin), 0), length), reverse=False)
    _apply_side(min(max(int(right_margin), 0), length), reverse=True)
    return weights


def _softmax_closeness_2d(
    height: int,
    width: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
    margin: int,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        return torch.zeros((height, width), dtype=torch.float32)

    if margin <= 0:
        return torch.zeros((height, width), dtype=torch.float32)

    margin_f = float(margin)
    large_distance = float(max(margin, height, width) + 1)

    rows = torch.arange(height, dtype=torch.float32)
    cols = torch.arange(width, dtype=torch.float32)

    if top_margin > 0:
        dist_top = rows
    else:
        dist_top = torch.full_like(rows, large_distance)
    if bottom_margin > 0:
        dist_bottom = (height - 1) - rows
    else:
        dist_bottom = torch.full_like(rows, large_distance)
    vertical = torch.minimum(dist_top, dist_bottom)

    if left_margin > 0:
        dist_left = cols
    else:
        dist_left = torch.full_like(cols, large_distance)
    if right_margin > 0:
        dist_right = (width - 1) - cols
    else:
        dist_right = torch.full_like(cols, large_distance)
    horizontal = torch.minimum(dist_left, dist_right)

    dist = torch.minimum(vertical.unsqueeze(1), horizontal.unsqueeze(0))
    clipped = torch.clamp(dist, max=margin_f)
    closeness = margin_f - clipped
    closeness.clamp_(min=0.0)
    return closeness


def make_spatial_blend_mask(
    window_shape: Tuple[int, int, int],
    sample_shape: Tuple[int, int, int],
    window_slice: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    margin: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    policy: str,
    min_value: float = 1e-3,
    tukey_alpha: float = 0.5,
    side_margins: Tuple[int, int, int, int] | None = None,
) -> SpatialBlendMaskResult:
    wt, wh, ww = map(int, window_shape)
    _, sample_h, sample_w = map(int, sample_shape)
    (_, _), (top, bottom), (left, right) = window_slice

    margin = max(int(margin), 0)

    if side_margins is not None:
        top_override, bottom_override, left_override, right_override = (
            max(int(side_margins[0]), 0),
            max(int(side_margins[1]), 0),
            max(int(side_margins[2]), 0),
            max(int(side_margins[3]), 0),
        )
        top_margin = min(top_override, wh)
        bottom_margin = min(bottom_override, wh)
        left_margin = min(left_override, ww)
        right_margin = min(right_override, ww)
        base_margin_h = max(top_margin, bottom_margin)
        base_margin_w = max(left_margin, right_margin)
    else:
        top_margin = min(margin, top)
        bottom_margin = min(margin, max(sample_h - bottom, 0))
        left_margin = min(margin, left)
        right_margin = min(margin, max(sample_w - right, 0))
        base_margin_h = margin
        base_margin_w = margin

    if policy == "softmax":
        cache_key = (
            wh,
            ww,
            top_margin,
            bottom_margin,
            left_margin,
            right_margin,
            margin,
        )
        closeness = _SOFTMAX_CACHE.get(cache_key)
        if closeness is None:
            closeness = _softmax_closeness_2d(
                wh,
                ww,
                top_margin,
                bottom_margin,
                left_margin,
                right_margin,
                margin,
            )
            _SOFTMAX_CACHE[cache_key] = closeness
        if wt > 1:
            closeness = closeness.unsqueeze(0).expand(wt, -1, -1)
        else:
            closeness = closeness.unsqueeze(0)
        closeness = closeness.to(device=device, dtype=dtype)
        return SpatialBlendMaskResult(mask=None, closeness=closeness, policy="softmax")

    cache_key = (
        policy,
        wh,
        ww,
        top_margin,
        bottom_margin,
        left_margin,
        right_margin,
        margin,
        float(min_value),
        float(tukey_alpha),
    )
    mask = _MASK_CACHE.get(cache_key)
    if mask is None:
        if policy == "hann":
            h_weights = _hann_edge_weights_1d(
                wh,
                base_margin_h,
                top_margin,
                bottom_margin,
                min_value=min_value,
            )
            w_weights = _hann_edge_weights_1d(
                ww,
                base_margin_w,
                left_margin,
                right_margin,
                min_value=min_value,
            )
        elif policy == "tukey":
            h_weights = _tukey_edge_weights_1d(
                wh,
                base_margin_h,
                top_margin,
                bottom_margin,
                alpha=tukey_alpha,
                min_value=min_value,
            )
            w_weights = _tukey_edge_weights_1d(
                ww,
                base_margin_w,
                left_margin,
                right_margin,
                alpha=tukey_alpha,
                min_value=min_value,
            )
        else:
            raise ValueError(f"Unsupported spatial blend policy for mask creation: {policy}")

        mask_2d = torch.outer(h_weights, w_weights)
        if wt > 1:
            mask = mask_2d.unsqueeze(0).expand(wt, -1, -1)
        else:
            mask = mask_2d.unsqueeze(0)
        _MASK_CACHE[cache_key] = mask

    if mask.device != torch.device("cpu"):
        mask = mask.to("cpu")
    mask = mask.to(device=device, dtype=dtype)
    return SpatialBlendMaskResult(mask=mask, closeness=None, policy=policy)

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
    #cal windows under 720p
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
    #cal windows under 720p
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
