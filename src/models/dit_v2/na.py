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

from itertools import chain
from typing import Callable, Dict, List, Sequence, Tuple
import einops
import torch

from .window import get_spatial_blend_settings, make_spatial_blend_mask


def flatten(
    hid: List[torch.FloatTensor],  # List of (*** c)
) -> Tuple[
    torch.FloatTensor,  # (L c)
    torch.LongTensor,  # (b n)
]:
    assert len(hid) > 0
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid[0].device) for x in hid])
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape


def unflatten(
    hid: torch.FloatTensor,  # (L c) or (L ... c)
    hid_shape: torch.LongTensor,  # (b n)
) -> List[torch.Tensor]:  # List of (*** c) or (*** ... c)
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    hid = [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]
    return hid


def concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> torch.FloatTensor:  # (L ... c)
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    return torch.cat(list(chain(*zip(vid, txt))))


def concat_idx(
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    Callable,
    Callable,
]:
    device = vid_len.device
    vid_idx = torch.arange(vid_len.sum(), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=device)
    tgt_idx = concat(vid_idx, txt_idx, vid_len, txt_len)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda vid, txt: torch.index_select(torch.cat([vid, txt]), 0, tgt_idx),
        lambda all: torch.index_select(all, 0, src_idx).split([len(vid_idx), len(txt_idx)]),
    )


def unconcat(
    all: torch.FloatTensor,  # (L ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    torch.FloatTensor,  # (VL ... c)
    torch.FloatTensor,  # (TL ... c)
]:
    interleave_len = list(chain(*zip(vid_len.tolist(), txt_len.tolist())))
    all = all.split(interleave_len)
    vid = torch.cat(all[0::2])
    txt = torch.cat(all[1::2])
    return vid, txt


def repeat_concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: List,  # (n)
) -> torch.FloatTensor:  # (L ... c)
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    txt = [[x] * n for x, n in zip(txt, txt_repeat)]
    txt = list(chain(*txt))
    return torch.cat(list(chain(*zip(vid, txt))))


def repeat_concat_idx(
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: torch.LongTensor,  # (n)
) -> Tuple[
    Callable,
    Callable,
]:
    device = vid_len.device
    vid_idx = torch.arange(vid_len.sum(), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=device)
    txt_repeat_list = txt_repeat.tolist()
    tgt_idx = repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    repeat_txt_len = (txt_len * txt_repeat).tolist()

    def unconcat_coalesce(all):
        """
        Un-concat vid & txt, and coalesce the repeated txt.
        e.g. vid [0 1 2 3 4 5 6 7 8] -> 3 splits -> [0 1 2] [3 4 5] [6 7 8]
             txt [9 10]
             repeat_concat ==> [0 1 2 9 10 3 4 5 9 10 6 7 8 9 10]
             1. argsort re-index ==> [0 1 2 3 4 5 6 7 8 9 9 9 10 10 10]
                           split ==> vid_out [0 1 2 3 4 5 6 7 8] txt_out [9 9 9 10 10 10]
             2. reshape & mean for each sample to coalesce the repeated txt.
        """
        vid_out, txt_out = all[src_idx].split([len(vid_idx), txt_idx_len])
        txt_out_coalesced = []
        for txt, repeat_time in zip(txt_out.split(repeat_txt_len), txt_repeat_list):
            txt = txt.reshape(-1, repeat_time, *txt.shape[1:]).mean(1)
            txt_out_coalesced.append(txt)
        return vid_out, torch.cat(txt_out_coalesced)

    # Note: Backward of torch.index_select is non-deterministic when existing repeated index,
    # the difference may cumulative like torch.repeat_interleave, so we use vanilla index here.
    return (
        lambda vid, txt: torch.cat([vid, txt])[tgt_idx],
        lambda all: unconcat_coalesce(all),
    )


def rearrange(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    return flatten([einops.rearrange(h, pattern, **kwargs) for h in unflatten(hid, hid_shape)])


def rearrange_idx(
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[Callable, Callable, torch.LongTensor]:
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape = rearrange(hid_idx, hid_shape, pattern, **kwargs)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: torch.index_select(hid, 0, src_idx),
        tgt_shape,
    )


def repeat(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, torch.LongTensor],  # (b)
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    hid = unflatten(hid, hid_shape)
    kwargs = [{k: v[i].item() for k, v in kwargs.items()} for i in range(len(hid))]
    return flatten([einops.repeat(h, pattern, **a) for h, a in zip(hid, kwargs)])


def pack(
    samples: List[torch.Tensor],  # List of (h w c).
) -> Tuple[
    List[torch.Tensor],  # groups [(b1 h1 w1 c1), (b2 h2 w2 c2)]
    List[List[int]],  # reversal indices.
]:
    batches = {}
    indices = {}
    for i, sample in enumerate(samples):
        shape = sample.shape
        batches[shape] = batches.get(shape, [])
        indices[shape] = indices.get(shape, [])
        batches[shape].append(sample)
        indices[shape].append(i)

    batches = list(map(torch.stack, batches.values()))
    indices = list(indices.values())
    return batches, indices


def unpack(
    batches: List[torch.Tensor],
    indices: List[List[int]],
) -> List[torch.Tensor]:
    samples = [None] * (max(chain(*indices)) + 1)
    for batch, index in zip(batches, indices):
        for sample, i in zip(batch.unbind(), index):
            samples[i] = sample
    return samples


def window(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    hid = unflatten(hid, hid_shape)
    hid = list(map(window_fn, hid))
    hid_windows = torch.tensor(list(map(len, hid)), device=hid_shape.device)
    hid, hid_shape = flatten(list(chain(*hid)))
    return hid, hid_shape, hid_windows


def window_idx(
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows = window(hid_idx, hid_shape, window_fn)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    window_slices = getattr(window_fn, "window_slices", None)
    halo_offsets_attr = getattr(window_fn, "window_halo_offsets", None)

    if isinstance(window_slices, Sequence):
        flat_slices: List[Tuple[slice, slice, slice]] = []
        sample_indices: List[int] = []
        flat_halo_offsets: List[Tuple[int, int, int, int]] = []
        halo_valid = isinstance(halo_offsets_attr, Sequence)
        for sample_idx, per_sample in enumerate(window_slices):
            if not isinstance(per_sample, Sequence):
                flat_slices = []
                sample_indices = []
                flat_halo_offsets = []
                halo_valid = False
                break
            halo_seq = (
                halo_offsets_attr[sample_idx]
                if halo_valid and sample_idx < len(halo_offsets_attr)
                else None
            )
            if halo_seq is not None and not isinstance(halo_seq, Sequence):
                halo_valid = False
                flat_halo_offsets = []
                halo_seq = None
            for window_idx_entry, slc in enumerate(per_sample):
                flat_slices.append(tuple(slc))
                sample_indices.append(sample_idx)
                if halo_valid and halo_seq is not None and window_idx_entry < len(halo_seq):
                    offsets = halo_seq[window_idx_entry]
                    if (
                        isinstance(offsets, Sequence)
                        and len(offsets) == 4
                        and all(isinstance(v, (int, float)) for v in offsets)
                    ):
                        flat_halo_offsets.append(
                            (int(offsets[0]), int(offsets[1]), int(offsets[2]), int(offsets[3]))
                        )
                    else:
                        halo_valid = False
                        flat_halo_offsets = []
                        halo_seq = None
                elif halo_valid:
                    halo_valid = False
                    flat_halo_offsets = []
                    halo_seq = None
        if len(flat_slices) != len(tgt_shape):
            flat_slices = []
            sample_indices = []
            flat_halo_offsets = []
        if not halo_valid or len(flat_halo_offsets) != len(flat_slices):
            flat_halo_offsets = []
    else:
        flat_slices = []
        sample_indices = []
        flat_halo_offsets = []

    has_slices = bool(flat_slices)
    if has_slices:
        sample_shapes = hid_shape.clone()
        window_shapes = tgt_shape.clone()
        sample_lookup = sample_indices.copy()
        slice_lookup = flat_slices.copy()
        halo_lookup = flat_halo_offsets.copy() if flat_halo_offsets else None
    else:
        sample_shapes = None
        window_shapes = None
        sample_lookup = None
        slice_lookup = None
        halo_lookup = None

    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: _window_reverse(
            hid,
            src_idx,
            tgt_idx,
            window_shapes,
            sample_shapes,
            slice_lookup,
            sample_lookup,
            halo_lookup,
        ),
        tgt_shape,
        tgt_windows,
    )


def _window_reverse(
    hid: torch.Tensor,
    src_idx: torch.LongTensor,
    tgt_idx: torch.LongTensor,
    window_shapes: torch.LongTensor | None,
    sample_shapes: torch.LongTensor | None,
    slice_lookup: List[Tuple[slice, slice, slice]] | None,
    sample_lookup: List[int] | None,
    halo_lookup: List[Tuple[int, int, int, int]] | None,
) -> torch.Tensor:
    policy, margin = get_spatial_blend_settings()
    if (
        policy != "hann"
        or margin <= 0
        or window_shapes is None
        or sample_shapes is None
        or not slice_lookup
        or sample_lookup is None
        or halo_lookup is None
        or hid.numel() == 0
    ):
        return torch.index_select(hid, 0, src_idx)

    feature_shape = hid.shape[1:]
    hid_flat = hid.reshape(hid.shape[0], -1)
    orig_len = sample_shapes.prod(-1).sum().item()

    out_flat = hid_flat.new_zeros((orig_len, hid_flat.shape[1]))
    weight_flat = hid_flat.new_zeros(orig_len)

    window_sizes = window_shapes.prod(-1).tolist()
    chunks = hid_flat.split(window_sizes, dim=0)
    idx_chunks = tgt_idx.split(window_sizes)

    def _slice_bounds(s: slice, length: int, total: int) -> Tuple[int, int]:
        start = 0 if s.start is None else int(s.start)
        if s.stop is None:
            stop = min(start + length, total)
        else:
            stop = int(s.stop)
        return start, stop

    for idx, (chunk, idx_vec) in enumerate(zip(chunks, idx_chunks)):
        sample_idx = sample_lookup[idx]
        sample_shape = tuple(int(v) for v in sample_shapes[sample_idx].tolist())
        win_shape = tuple(int(v) for v in window_shapes[idx].tolist())
        slc_t, slc_h, slc_w = slice_lookup[idx]
        norm_slice = (
            _slice_bounds(slc_t, win_shape[0], sample_shape[0]),
            _slice_bounds(slc_h, win_shape[1], sample_shape[1]),
            _slice_bounds(slc_w, win_shape[2], sample_shape[2]),
        )
        halo_offsets = halo_lookup[idx]
        side_margins = (
            min(margin, max(int(halo_offsets[0]), 0)),
            min(margin, max(int(halo_offsets[1]), 0)),
            min(margin, max(int(halo_offsets[2]), 0)),
            min(margin, max(int(halo_offsets[3]), 0)),
        )
        if side_margins == (0, 0, 0, 0):
            side_margins = None
        mask = make_spatial_blend_mask(
            win_shape,
            sample_shape,
            norm_slice,
            margin,
            device=hid.device,
            dtype=hid.dtype,
            side_margins=side_margins,
        )
        mask_flat = mask.reshape(-1)
        weighted = chunk * mask_flat.unsqueeze(-1)
        out_flat.index_add_(0, idx_vec, weighted)
        weight_flat.index_add_(0, idx_vec, mask_flat)

    out_flat = out_flat / torch.clamp(weight_flat.unsqueeze(-1), min=1e-6)
    return out_flat.reshape((orig_len,) + feature_shape)
