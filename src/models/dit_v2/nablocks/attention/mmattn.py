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

from typing import Dict, Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from .....common.cache import Cache
from .....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv
from .....common.half_precision_fixes import safe_pad_operation

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import WindowPlan, compute_adaptive_windows, get_window_op
from .....utils.window_logging import WindowLogger, plan_from_slices
from itertools import chain


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen()

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)

        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        attn = self.attn(
            q=concat(vid_q, txt_q).bfloat16(),
            k=concat(vid_k, txt_k).bfloat16(),
            v=concat(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache("mm_maxlen", lambda: all_len.max().item()),
            max_seqlen_k=cache("mm_maxlen", lambda: all_len.max().item()),
        ).type_as(vid_q)

        attn = rearrange(attn, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        adaptive_windows: bool = False,
        rope_apply_global: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)
        self._log_regular_window_op = get_window_op("720pwin_by_size_bysize")
        self._log_shifted_window_op = get_window_op("720pswin_by_size_bysize")
        self.window_logger: Optional[WindowLogger] = None
        self.adaptive_windows = adaptive_windows
        self.rope_apply_global = rope_apply_global
        self._rope_default_policy = bool(rope_apply_global)
        self._rope_logged = False
        self._window_plan_cache: Dict[
            Tuple[int, int, int, bool, torch.device, torch.dtype],
            WindowPlan,
        ] = {}

    def set_runtime_window_flags(
        self,
        *,
        adaptive_windows: Optional[bool] = None,
        rope_apply_global_override: Optional[bool] = None,
        rope_apply_global: Optional[bool] = None,
    ) -> None:
        cache_invalidated = False
        if adaptive_windows is not None and adaptive_windows != self.adaptive_windows:
            self.adaptive_windows = adaptive_windows
            cache_invalidated = True
        override = (
            rope_apply_global_override
            if rope_apply_global_override is not None
            else rope_apply_global
        )
        desired_rope = self._rope_default_policy if override is None else bool(override)
        if desired_rope != self.rope_apply_global:
            self.rope_apply_global = desired_rope
            self._rope_logged = False
        if cache_invalidated:
            self._window_plan_cache.clear()

    def _get_adaptive_plan(
        self,
        dt: int,
        dh: int,
        dw: int,
        device: torch.device,
        dtype: torch.dtype,
        is_shifted: bool,
    ) -> WindowPlan:
        key = (dt, dh, dw, is_shifted, device, dtype)
        plan = self._window_plan_cache.get(key)
        if plan is None:
            target_nh = max(1, int(self.window[1]))
            target_nw = max(1, int(self.window[2]))
            plan = compute_adaptive_windows(
                dt,
                dh,
                dw,
                target_nh=target_nh,
                target_nw=target_nw,
            )
            base_key = (dt, dh, dw, False, device, dtype)
            shifted_key = (dt, dh, dw, True, device, dtype)
            self._window_plan_cache[base_key] = plan
            self._window_plan_cache[shifted_key] = plan
        return plan

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        window_logger = getattr(self, "window_logger", None)
        latent_shape = tuple(int(v) for v in vid_shape[0].tolist())
        dt, dh, dw = latent_shape
        is_shifted = self.window_method == "720pswin_by_size_bysize"

        if (
            window_logger is not None
            and getattr(window_logger, "_latest_plan", None) is None
        ):
            self._rope_logged = False

        if (
            window_logger is not None
            and getattr(window_logger, "config", None)
            and getattr(window_logger.config, "log_window_info", False)
            and not self._rope_logged
        ):
            rope_impl = type(self.rope).__name__ if self.rope is not None else "None"
            policy = "global_prewindow" if self.rope_apply_global else "per_window_restart"
            print(f"[ROPE] policy={policy} impl={rope_impl}")
            self._rope_logged = True

        if self.adaptive_windows:
            plan = self._get_adaptive_plan(dt, dh, dw, vid.device, vid.dtype, is_shifted)
            regular_slices = list(plan.slices.regular)
            shifted_slices = list(plan.slices.shifted)
            slices_for_variant = shifted_slices if is_shifted else regular_slices
            if window_logger is not None and window_logger.enabled:
                plan_dump = plan_from_slices(latent_shape, regular_slices, shifted_slices)
                window_logger.process(
                    "adaptive",
                    latent_shape,
                    regular_slices,
                    shifted_slices,
                    plan=plan_dump,
                    proxy_hw=(plan.proxy_h, plan.proxy_w),
                )
        else:
            base_window = tuple(int(v) for v in self.window)
            regular_slices = list(self._log_regular_window_op(latent_shape, base_window))
            shifted_slices = list(self._log_shifted_window_op(latent_shape, base_window))
            slices_for_variant = list(self.window_op(latent_shape, base_window))
            if window_logger is not None and window_logger.enabled:
                window_logger.process("fixed", latent_shape, regular_slices, shifted_slices)

        slices_for_variant = list(slices_for_variant)

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )

        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope and self.rope_apply_global:
            cache_global = cache.namespace("global_rope")
            if getattr(self.rope, "mm", False):
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q,
                    vid_k,
                    vid_shape,
                    txt_q,
                    txt_k,
                    txt_shape,
                    cache_global,
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache_global)

        concat_vid = torch.stack((vid_q, vid_k, vid_v), dim=1)
        concat_vid = rearrange(concat_vid, "l o h d -> l (o h d)")

        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor, slices=slices_for_variant):
            return [x[st, sh, sw] for (st, sh, sw) in slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(concat_vid)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)

        if self.rope and not self.rope_apply_global:
            if getattr(self.rope, "mm", False):
                _, num_h, _ = txt_q.shape
                txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
                txt_q_repeat = na.unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [[x] * n for x, n in zip(txt_q_repeat, window_count)]
                txt_q_repeat = list(chain(*txt_q_repeat))
                txt_q_repeat, txt_shape_repeat = na.flatten(txt_q_repeat)
                txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)

                txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
                txt_k_repeat = na.unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [[x] * n for x, n in zip(txt_k_repeat, window_count)]
                txt_k_repeat = list(chain(*txt_k_repeat))
                txt_k_repeat, _ = na.flatten(txt_k_repeat)
                txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q,
                    vid_k,
                    window_shape,
                    txt_q_repeat,
                    txt_k_repeat,
                    txt_shape_repeat,
                    cache_win,
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )
            
        out = self.attn(
            q=concat_win(vid_q, txt_q).bfloat16(),
            k=concat_win(vid_k, txt_k).bfloat16(),
            v=concat_win(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache_win(
                "vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            cu_seqlens_k=cache_win(
                "vid_seqlens_k", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max().item()),
            max_seqlen_k=cache_win("vid_max_seqlen_k", lambda: all_len_win.max().item()),
        ).type_as(vid_q)

        # text pooling
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out