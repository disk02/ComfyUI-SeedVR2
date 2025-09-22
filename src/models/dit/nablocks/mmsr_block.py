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

from typing import Tuple, Union
import math
import torch
from einops import rearrange
from torch.nn import functional as F

# from ..cache import Cache
from ....common.cache import Cache
from ....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from .. import na
from ..attention import FlashAttentionVarlen
from ..blocks.mmdit_window_block import MMWindowAttention, MMWindowTransformerBlock
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import NaRotaryEmbedding3d
from ..window import get_window_op
from ....common.half_precision_fixes import safe_pad_operation

class NaSwinAttention(MMWindowAttention):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        log_window_info: bool = False,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
        )
        self.rope = NaRotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = FlashAttentionVarlen()
        self.window_op = get_window_op(window_method)
        self._log_window_info = bool(log_window_info)
        self._did_log_banner = False
        self._lattice_cache = []
        self._layer_index = None
        self._attention_variant = "dit"

    @staticmethod
    def _compute_lattice(latent_shape, base_window, shifted=False):
        t, h, w = latent_shape
        base_nt, base_nh, base_nw = base_window
        resized_nt = base_nt if base_nt else 1
        resized_nh = base_nh if base_nh else 1
        resized_nw = base_nw if base_nw else 1
        safe_hw = max(h * w, 1)
        scale = math.sqrt((45 * 80) / safe_hw)
        resized_h = round(h * scale)
        resized_w = round(w * scale)
        wh = math.ceil(resized_h / resized_nh) if resized_nh else h
        ww = math.ceil(resized_w / resized_nw) if resized_nw else w
        wt = math.ceil(min(t, 30) / resized_nt) if resized_nt else max(t, 1)
        wt = max(wt, 1)
        wh = max(wh, 1)
        ww = max(ww, 1)

        if shifted:
            st = 0.5 if wt < t else 0.0
            sh = 0.5 if wh < h else 0.0
            sw = 0.5 if ww < w else 0.0
            nt = math.ceil((t - st) / wt) if wt else 0
            nh = math.ceil((h - sh) / wh) if wh else 0
            nw = math.ceil((w - sw) / ww) if ww else 0
            nt = nt + 1 if st > 0 else 1
            nh = nh + 1 if sh > 0 else 1
            nw = nw + 1 if sw > 0 else 1
        else:
            nt = math.ceil(t / wt) if wt else 0
            nh = math.ceil(h / wh) if wh else 0
            nw = math.ceil(w / ww) if ww else 0

        return (int(wt), int(wh), int(ww)), (int(nt), int(nh), int(nw))

    def _maybe_log_lattice(self, vid_shape: torch.LongTensor):
        if not self._log_window_info or self._did_log_banner:
            return

        latent_dims = tuple(int(x) for x in vid_shape[0].tolist())
        regular_p, regular_n = self._compute_lattice(latent_dims, self.window, shifted=False)
        shifted_p, shifted_n = self._compute_lattice(latent_dims, self.window, shifted=True)
        module_label = f"{self.__class__.__name__}(layer={self._layer_index if self._layer_index is not None else 'NA'}, variant={self._attention_variant})"
        latent_str = f"[{latent_dims[0]},{latent_dims[1]},{latent_dims[2]}]"
        regular_p_str = f"[{regular_p[0]},{regular_p[1]},{regular_p[2]}]"
        regular_n_str = f"[{regular_n[0]},{regular_n[1]},{regular_n[2]}]"
        shifted_p_str = f"[{shifted_p[0]},{shifted_p[1]},{shifted_p[2]}]"
        shifted_n_str = f"[{shifted_n[0]},{shifted_n[1]},{shifted_n[2]}]"
        print(
            f"[ATTN] ATTN_MODE=fixed latent={latent_str} "
            f"regular p={regular_p_str} n={regular_n_str} "
            f"shifted p={shifted_p_str} n={shifted_n_str} module={module_label}"
        )
        self._lattice_cache.append(
            {
                "latent": [int(x) for x in latent_dims],
                "regular": {"p": [int(x) for x in regular_p], "n": [int(x) for x in regular_n]},
                "shifted": {"p": [int(x) for x in shifted_p], "n": [int(x) for x in shifted_n]},
                "module": module_label,
            }
        )
        self._did_log_banner = True

    def set_layer_index(self, layer_index: int, attention_variant: str = "dit"):
        self._layer_index = layer_index
        self._attention_variant = attention_variant or self._attention_variant

    def pop_window_lattice(self):
        data, self._lattice_cache = self._lattice_cache, []
        return data

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

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        self._maybe_log_lattice(vid_shape)

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        # window rope
        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

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


class NaMMSRTransformerBlock(MMWindowTransformerBlock):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        **kwargs,
    ):
        log_window_info = kwargs.pop("log_window_info", False)
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            emb_dim=emb_dim,
            heads=heads,
            head_dim=head_dim,
            expand_ratio=expand_ratio,
            norm=norm,
            norm_eps=norm_eps,
            ada=ada,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            shared_qkv=shared_qkv,
            shared_mlp=shared_mlp,
            mlp_type=mlp_type,
            **kwargs,
        )

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            shared_qkv=shared_qkv,
            log_window_info=log_window_info,
            **kwargs,
        )

    def set_layer_index(self, layer_index: int, attention_variant: str = "dit"):
        if hasattr(self.attn, "set_layer_index"):
            self.attn.set_layer_index(layer_index, attention_variant)

    def pop_window_lattice(self):
        if hasattr(self.attn, "pop_window_lattice"):
            return self.attn.pop_window_lattice()
        return []

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape
