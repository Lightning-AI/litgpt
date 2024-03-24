# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch
from .utils import calculate_settings

ROPE_GROUP_SIZE = 4

@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit
def _rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1
    pass

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% Faster kernel from [HuyNguyen-hust](https://github.com/unslothai/unsloth/pull/238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim

        # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)
    pass
pass


def _rope_embedding_forward_impl(Q, cos, sin):
    Q = Q.transpose(1, 2).clone()
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, seq_len, n_heads, head_dim = Q.shape
    Q = Q.reshape(batch*seq_len, n_heads*head_dim)
    n_rows, n_cols = Q.shape
    assert(seq_len <= cos.shape[0])

    # [TODO] Changing blocksize to head_dim//2 seems to have
    # some concurrency / un-deterministic issues.
    BLOCK_SIZE, num_warps = calculate_settings(head_dim//2) # (head_dim//2)

    # group_size = 4 # 4 or 8, too large group_size can hurt performance.
    div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)

    _rope_embedding[(n_rows, n_groups, )](
          Q,   Q.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seq_len,
        head_dim, n_heads,
        BACKWARD_PASS = False,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    Q = Q.view(batch, seq_len, n_heads, head_dim)
    Q = Q.transpose(1, 2)
    return Q, cos, sin, n_groups, BLOCK_SIZE, num_warps


def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    dY = dY.transpose(1, 2)
    batch, seq_len, n_heads, head_dim = dY.shape
    dY = dY.reshape(batch*seq_len, n_heads*head_dim)
    # Must be reshape not view
    n_rows, n_cols = dY.shape

    _rope_embedding[(n_rows, n_groups, )](
        dY,  dY .stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seq_len, head_dim, n_heads,
        BACKWARD_PASS = True,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    dY = dY.view(batch, seq_len, n_heads, head_dim)
    dY = dY.transpose(1, 2)
    return dY