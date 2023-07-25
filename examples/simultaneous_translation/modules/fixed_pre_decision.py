from functools import partial

import torch
from torch import Tensor
import math
import torch.nn.functional as F

from . import register_monotonic_attention
from .monotonic_multihead_attention import (
    MonotonicMultiheadAttentionWaitK,
    MonotonicMultiheadAttentionHardAligned,
    MonotonicMultiheadAttentionInfiniteLookback,
)
from typing import Dict, Optional
import random


def fixed_pooling_monotonic_attention(monotonic_attention):
    def create_model(monotonic_attention, klass):
        class FixedStrideMonotonicAttention(monotonic_attention):
            def __init__(self, args):
                super().__init__(args)
                self.pre_decision_type = args.fixed_pre_decision_type
                self.pre_decision_pad_threshold = args.fixed_pre_decision_pad_threshold
                if hasattr(args, "multi_path_stride") and args.multi_path_stride:
                    self.pre_decision_ratio = random.randint(
                        args.fixed_pre_decision_ratio_min, args.fixed_pre_decision_ratio
                    )
                else:
                    self.pre_decision_ratio = args.fixed_pre_decision_ratio
                if self.pre_decision_ratio == 1:
                    return

                if args.fixed_pre_decision_type == "average":
                    self.pooling_layer = torch.nn.AvgPool1d(
                        kernel_size=self.pre_decision_ratio,
                        stride=self.pre_decision_ratio,
                        ceil_mode=True,
                    )
                elif args.fixed_pre_decision_type == "last":

                    def last(key):
                        if key.size(2) < self.pre_decision_ratio:
                            return key
                        else:
                            k = key[
                                :,
                                :,
                                self.pre_decision_ratio - 1 :: self.pre_decision_ratio,
                            ].contiguous()
                            if key.size(-1) % self.pre_decision_ratio != 0:
                                k = torch.cat([k, key[:, :, -1:]], dim=-1).contiguous()
                            return k

                    self.pooling_layer = last
                else:
                    raise NotImplementedError

            @staticmethod
            def add_args(parser):
                super(
                    FixedStrideMonotonicAttention, FixedStrideMonotonicAttention
                ).add_args(parser)
                parser.add_argument(
                    "--fixed-pre-decision-ratio",
                    type=int,
                    required=True,
                    help=(
                        "Ratio for the fixed pre-decision,"
                        "indicating how many encoder steps will start"
                        "simultaneous decision making process."
                    ),
                )
                parser.add_argument(
                    "--fixed-pre-decision-type",
                    default="average",
                    choices=["average", "last"],
                    help="Pooling type",
                )
                parser.add_argument(
                    "--fixed-pre-decision-pad-threshold",
                    type=float,
                    default=0.3,
                    help="If a part of the sequence has pad"
                    ",the threshold the pooled part is a pad.",
                )
                parser.add_argument(
                    "--multi-path-stride",
                    type=bool,
                    default=False,
                    help="whether using the multi-path strategy on stride",
                )
                parser.add_argument(
                    "--fixed-pre-decision-ratio-min",
                    type=int,
                    default=1,
                    help="the mimmum ratio for multi-path-stride",
                )

            def insert_zeros(self, x):
                bsz_num_heads, tgt_len, src_len = x.size()
                stride = self.pre_decision_ratio
                weight = F.pad(torch.ones(1, 1, 1).to(x), (stride - 1, 0))
                x_upsample = F.conv_transpose1d(
                    x.view(-1, src_len).unsqueeze(1),
                    weight,
                    stride=stride,
                    padding=0,
                )
                return x_upsample.squeeze(1).view(bsz_num_heads, tgt_len, -1)

            def p_choose_waitk(
                self,
                query,
                key,
                key_padding_mask: Optional[Tensor] = None,
                incremental_state: Optional[
                    Dict[str, Dict[str, Optional[Tensor]]]
                ] = None,
            ):
                """
                query: bsz, tgt_len
                key: bsz, src_len
                key_padding_mask: bsz, src_len
                """
                if incremental_state is not None:
                    # Retrieve target length from incremental states
                    # For inference the length of query is always 1
                    tgt = incremental_state["steps"]["tgt"]
                    assert tgt is not None
                    tgt_len = int(tgt)
                else:
                    tgt_len, bsz, _ = query.size()

                src_len, bsz, _ = key.size()

                p_choose = torch.ones(bsz, tgt_len, src_len).to(query)
                p_choose = torch.tril(p_choose, diagonal=self.waitk_lagging - 1)
                p_choose = torch.triu(p_choose, diagonal=self.waitk_lagging - 1)

                if incremental_state is not None:
                    p_choose = p_choose[:, -1:]
                    tgt_len = 1

                # Extend to each head
                p_choose = (
                    p_choose.contiguous()
                    .unsqueeze(1)
                    .expand(-1, self.num_heads, -1, -1)
                    .contiguous()
                    .view(-1, tgt_len, src_len)
                )

                return p_choose

            def p_choose(
                self,
                query: Optional[Tensor],
                key: Optional[Tensor],
                key_padding_mask: Optional[Tensor] = None,
                incremental_state: Optional[
                    Dict[str, Dict[str, Optional[Tensor]]]
                ] = None,
            ):
                assert key is not None
                assert query is not None
                src_len = key.size(0)
                tgt_len = query.size(0)
                batch_size = query.size(1)

                if self.pre_decision_ratio == 1:
                    return self.p_choose_waitk(
                        query,
                        key,
                        key_padding_mask,
                        incremental_state=incremental_state,
                    )

                key_pool = self.pooling_layer(key.transpose(0, 2)).transpose(0, 2)

                if key_padding_mask is not None:
                    key_padding_mask_pool = (
                        self.pooling_layer(key_padding_mask.unsqueeze(0).float())
                        .squeeze(0)
                        .gt(self.pre_decision_pad_threshold)
                    )
                    # Make sure at least one element is not pad
                    key_padding_mask_pool[:, 0] = 0
                else:
                    key_padding_mask_pool = None

                if incremental_state is not None:
                    # The floor instead of ceil is used for inference
                    # But make sure the length key_pool at least 1
                    if (
                        max(1, math.floor(key.size(0) / self.pre_decision_ratio))
                    ) < key_pool.size(0):
                        key_pool = key_pool[:-1]
                        if key_padding_mask_pool is not None:
                            key_padding_mask_pool = key_padding_mask_pool[:-1]

                p_choose_pooled = self.p_choose_waitk(
                    query,
                    key_pool,
                    key_padding_mask_pool,
                    incremental_state=incremental_state,
                )

                # Upsample, interpolate zeros
                p_choose = self.insert_zeros(p_choose_pooled)

                if p_choose.size(-1) < src_len:
                    # Append zeros if the upsampled p_choose is shorter than src_len
                    p_choose = torch.cat(
                        [
                            p_choose,
                            torch.zeros(
                                p_choose.size(0), tgt_len, src_len - p_choose.size(-1)
                            ).to(p_choose),
                        ],
                        dim=2,
                    )
                else:
                    # can be larger than src_len because we used ceil before
                    p_choose = p_choose[:, :, :src_len]
                    p_choose[:, :, -1] = p_choose_pooled[:, :, -1]

                assert list(p_choose.size()) == [
                    batch_size * self.num_heads,
                    tgt_len,
                    src_len,
                ]

                return p_choose

        FixedStrideMonotonicAttention.__name__ = klass.__name__
        return FixedStrideMonotonicAttention

    return partial(create_model, monotonic_attention)


def fixed_pooling_monotonic_attention_pchoose_mma(monotonic_attention):
    def create_model(monotonic_attention, klass):
        class FixedStrideMonotonicAttention(monotonic_attention):
            def __init__(self, args):
                super().__init__(args)
                self.pre_decision_type = args.fixed_pre_decision_type
                self.pre_decision_pad_threshold = args.fixed_pre_decision_pad_threshold

                if args.multi_path_stride:
                    self.pre_decision_ratio = random.randrange(
                        args.fixed_pre_decision_ratio_min,
                        args.fixed_pre_decision_ratio_max,
                    )
                else:
                    self.pre_decision_ratio = args.fixed_pre_decision_ratio
                if self.pre_decision_ratio == 1:
                    return

                if args.fixed_pre_decision_type == "average":
                    self.pooling_layer = torch.nn.AvgPool1d(
                        kernel_size=self.pre_decision_ratio,
                        stride=self.pre_decision_ratio,
                        ceil_mode=True,
                    )
                elif args.fixed_pre_decision_type == "last":

                    def last(key):
                        if key.size(2) < self.pre_decision_ratio:
                            return key
                        else:
                            k = key[
                                :,
                                :,
                                self.pre_decision_ratio - 1 :: self.pre_decision_ratio,
                            ].contiguous()
                            if key.size(-1) % self.pre_decision_ratio != 0:
                                k = torch.cat([k, key[:, :, -1:]], dim=-1).contiguous()
                            return k

                    self.pooling_layer = last
                else:
                    raise NotImplementedError

            @staticmethod
            def add_args(parser):
                super(
                    FixedStrideMonotonicAttention, FixedStrideMonotonicAttention
                ).add_args(parser)
                parser.add_argument(
                    "--fixed-pre-decision-ratio",
                    type=int,
                    default=7,
                    help=(
                        "Ratio for the fixed pre-decision,"
                        "indicating how many encoder steps will start"
                        "simultaneous decision making process."
                    ),
                )
                parser.add_argument(
                    "--fixed-pre-decision-type",
                    default="average",
                    choices=["average", "last"],
                    help="Pooling type",
                )
                parser.add_argument(
                    "--fixed-pre-decision-pad-threshold",
                    type=float,
                    default=0.3,
                    help="If a part of the sequence has pad"
                    ",the threshold the pooled part is a pad.",
                )
                parser.add_argument(
                    "--multi-path-stride",
                    type=bool,
                    default=False,
                    help="whether using the multi-path strategy on stride",
                )
                parser.add_argument(
                    "--fixed-pre-decision-ratio-min",
                    type=int,
                    default=1,
                    help="the mimmum ratio for multi-path-stride",
                )
                parser.add_argument(
                    "--fixed-pre-decision-ratio-max",
                    type=int,
                    default=1000,
                    help="the maxmum ratio for multi-path-stride",
                )

            def insert_zeros(self, x):
                bsz_num_heads, tgt_len, src_len = x.size()
                stride = self.pre_decision_ratio
                weight = F.pad(torch.ones(1, 1, 1).to(x), (stride - 1, 0))
                x_upsample = F.conv_transpose1d(
                    x.view(-1, src_len).unsqueeze(1),
                    weight,
                    stride=stride,
                    padding=0,
                )
                return x_upsample.squeeze(1).view(bsz_num_heads, tgt_len, -1)

            def p_choose_waitk(
                self,
                query,
                key,
                key_padding_mask: Optional[Tensor] = None,
                incremental_state: Optional[
                    Dict[str, Dict[str, Optional[Tensor]]]
                ] = None,
            ):
                """
                query: bsz, tgt_len
                key: bsz, src_len
                key_padding_mask: bsz, src_len
                """
                if incremental_state is not None:
                    # Retrieve target length from incremental states
                    # For inference the length of query is always 1
                    tgt = incremental_state["steps"]["tgt"]
                    assert tgt is not None
                    tgt_len = int(tgt)
                else:
                    tgt_len, bsz, _ = query.size()

                src_len, bsz, _ = key.size()

                p_choose = torch.ones(bsz, tgt_len, src_len).to(query)
                p_choose = torch.tril(p_choose, diagonal=self.waitk_lagging - 1)
                p_choose = torch.triu(p_choose, diagonal=self.waitk_lagging - 1)

                if incremental_state is not None:
                    p_choose = p_choose[:, -1:]
                    tgt_len = 1

                # Extend to each head
                p_choose = (
                    p_choose.contiguous()
                    .unsqueeze(1)
                    .expand(-1, self.num_heads, -1, -1)
                    .contiguous()
                    .view(-1, tgt_len, src_len)
                )

                return p_choose

            def p_choose_mma(
                self,
                query: Optional[Tensor],
                key: Optional[Tensor],
                key_padding_mask: Optional[Tensor] = None,
                incremental_state: Optional[
                    Dict[str, Dict[str, Optional[Tensor]]]
                ] = None,
            ):
                """
                Calculating step wise prob for reading and writing
                1 to read, 0 to write

                ============================================================
                Expected input size
                query: bsz, tgt_len, embed_dim
                key: bsz, src_len, embed_dim
                value: bsz, src_len, embed_dim
                key_padding_mask: bsz, src_len
                attn_mask: bsz, src_len
                query: bsz, tgt_len, embed_dim
                """

                # prepare inputs
                q_proj, k_proj, _ = self.input_projections(
                    query, key, None, "monotonic"
                )

                # attention energy
                attn_energy = self.attn_energy(q_proj, k_proj, key_padding_mask)

                noise = 0

                if self.training:
                    # add noise here to encourage discretness
                    noise = (
                        torch.normal(
                            self.noise_mean, self.noise_var, attn_energy.size()
                        )
                        .type_as(attn_energy)
                        .to(attn_energy.device)
                    )

                p_choose = torch.sigmoid(attn_energy + noise)
                _, _, tgt_len, src_len = p_choose.size()

                # p_choose: bsz * self.num_heads, tgt_len, src_len
                return p_choose.view(-1, tgt_len, src_len)

            def p_choose(
                self,
                query: Optional[Tensor],
                key: Optional[Tensor],
                key_padding_mask: Optional[Tensor] = None,
                incremental_state: Optional[
                    Dict[str, Dict[str, Optional[Tensor]]]
                ] = None,
            ):
                assert key is not None
                assert query is not None
                src_len = key.size(0)
                tgt_len = query.size(0)
                batch_size = query.size(1)

                if self.pre_decision_ratio == 1:
                    return self.p_choose_mma(
                        query,
                        key,
                        key_padding_mask,
                        incremental_state=incremental_state,
                    )

                key_pool = self.pooling_layer(key.transpose(0, 2)).transpose(0, 2)

                if key_padding_mask is not None:
                    key_padding_mask_pool = (
                        self.pooling_layer(key_padding_mask.unsqueeze(0).float())
                        .squeeze(0)
                        .gt(self.pre_decision_pad_threshold)
                    )
                    # Make sure at least one element is not pad
                    key_padding_mask_pool[:, 0] = 0
                else:
                    key_padding_mask_pool = None

                if incremental_state is not None:
                    # The floor instead of ceil is used for inference
                    # But make sure the length key_pool at least 1
                    if (
                        max(1, math.floor(key.size(0) / self.pre_decision_ratio))
                    ) < key_pool.size(0):
                        key_pool = key_pool[:-1]
                        if key_padding_mask_pool is not None:
                            key_padding_mask_pool = key_padding_mask_pool[:-1]

                p_choose_pooled = self.p_choose_mma(
                    query,
                    key_pool,
                    key_padding_mask_pool,
                    incremental_state=incremental_state,
                )

                # Upsample, interpolate zeros
                p_choose = self.insert_zeros(p_choose_pooled)

                if p_choose.size(-1) < src_len:
                    # Append zeros if the upsampled p_choose is shorter than src_len
                    p_choose = torch.cat(
                        [
                            p_choose,
                            torch.zeros(
                                p_choose.size(0), tgt_len, src_len - p_choose.size(-1)
                            ).to(p_choose),
                        ],
                        dim=2,
                    )
                else:
                    # can be larger than src_len because we used ceil before
                    p_choose = p_choose[:, :, :src_len]
                    p_choose[:, :, -1] = p_choose_pooled[:, :, -1]

                assert list(p_choose.size()) == [
                    batch_size * self.num_heads,
                    tgt_len,
                    src_len,
                ]

                return p_choose

        FixedStrideMonotonicAttention.__name__ = klass.__name__
        return FixedStrideMonotonicAttention

    return partial(create_model, monotonic_attention)


@register_monotonic_attention("waitk_fixed_pre_decision")
@fixed_pooling_monotonic_attention(MonotonicMultiheadAttentionWaitK)
class MonotonicMultiheadAttentionWaitkFixedStride:
    pass


@register_monotonic_attention("hard_aligned_fixed_pre_decision")
@fixed_pooling_monotonic_attention(MonotonicMultiheadAttentionHardAligned)
class MonotonicMultiheadAttentionHardFixedStride:
    pass


@register_monotonic_attention("infinite_lookback_fixed_pre_decision")
@fixed_pooling_monotonic_attention(MonotonicMultiheadAttentionInfiniteLookback)
class MonotonicMultiheadAttentionInfiniteLookbackFixedStride:
    pass


@register_monotonic_attention("infinite_lookback_fixed_pre_decision_pchoose_mma")
@fixed_pooling_monotonic_attention_pchoose_mma(
    MonotonicMultiheadAttentionInfiniteLookback
)
class MonotonicMultiheadAttentionInfiniteLookbackFixedStridePchooseMMA:
    pass
