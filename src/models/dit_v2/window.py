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
from typing import Dict, Tuple
import math

import torch

_SPATIAL_BLEND_POLICY: str = "hann"
_SPATIAL_BLEND_MARGIN: int = 24
_HANN_CACHE: Dict[Tuple[int, int, int, int, int, int, torch.dtype, str], torch.Tensor] = {}


def set_spatial_blend(policy: str, margin: int) -> None:
    global _SPATIAL_BLEND_POLICY, _SPATIAL_BLEND_MARGIN
    if policy not in {"off", "hann"}:
        raise ValueError(f"Unsupported spatial blend policy: {policy}")
    _SPATIAL_BLEND_POLICY = policy
    _SPATIAL_BLEND_MARGIN = max(int(margin), 0)


def get_spatial_blend_settings() -> Tuple[str, int]:
    return _SPATIAL_BLEND_POLICY, _SPATIAL_BLEND_MARGIN


def _hann_edge_weights_1d(
    length: int,
    base_margin: int,
    left_margin: int,
    right_margin: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_value: float = 1e-3,
) -> torch.Tensor:
    if length <= 0:
        return torch.ones(0, device=device, dtype=dtype)

    weights = torch.ones(length, device=device, dtype=dtype)
    if base_margin <= 0:
        return weights

    left = min(max(int(left_margin), 0), length)
    right = min(max(int(right_margin), 0), length)

    if left > 0:
        ramp = torch.linspace(0.0, math.pi, left, device=device, dtype=torch.float32)
        ramp = min_value + (1.0 - min_value) * 0.5 * (1.0 - torch.cos(ramp))
        weights[:left] = ramp.to(dtype=dtype)
    if right > 0:
        ramp = torch.linspace(0.0, math.pi, right, device=device, dtype=torch.float32)
        ramp = min_value + (1.0 - min_value) * 0.5 * (1.0 - torch.cos(ramp))
        weights[-right:] = torch.flip(ramp.to(dtype=dtype), dims=[0])
    return weights


def make_spatial_blend_mask(
    window_shape: Tuple[int, int, int],
    sample_shape: Tuple[int, int, int],
    window_slice: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    margin: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_value: float = 1e-3,
) -> torch.Tensor:
    wt, wh, ww = map(int, window_shape)
    _, sample_h, sample_w = map(int, sample_shape)
    (_, _), (top, bottom), (left, right) = window_slice

    margin = max(int(margin), 0)

    top_margin = min(margin, top)
    bottom_margin = min(margin, max(sample_h - bottom, 0))
    left_margin = min(margin, left)
    right_margin = min(margin, max(sample_w - right, 0))

    cache_key = (
        wh,
        ww,
        top_margin,
        bottom_margin,
        left_margin,
        right_margin,
        dtype,
        str(device),
    )
    mask = _HANN_CACHE.get(cache_key)
    if mask is None:
        h_weights = _hann_edge_weights_1d(
            wh,
            margin,
            top_margin,
            bottom_margin,
            device=device,
            dtype=dtype,
            min_value=min_value,
        )
        w_weights = _hann_edge_weights_1d(
            ww,
            margin,
            left_margin,
            right_margin,
            device=device,
            dtype=dtype,
            min_value=min_value,
        )
        mask_2d = torch.outer(h_weights, w_weights)
        if wt > 1:
            mask = mask_2d.unsqueeze(0).expand(wt, -1, -1)
        else:
            mask = mask_2d.unsqueeze(0)
        _HANN_CACHE[cache_key] = mask
    if mask.device != device or mask.dtype != dtype:
        mask = mask.to(device=device, dtype=dtype)
    return mask

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
