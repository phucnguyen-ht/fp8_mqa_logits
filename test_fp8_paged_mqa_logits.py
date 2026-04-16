# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import argparse
import functools

import torch
import os
import pandas as pd

from aiter.test_common import run_perftest, checkAllclose
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from moreh_fp8_paged_mqa_logits import fp8_paged_mqa_logits as moreh_fp8_paged_mqa_logits


@functools.cache
def get_num_compute_units() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count

ATOL=RTOL=0.01

def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y

def kv_cache_cast_to_fp8(x: torch.Tensor, padding=False) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fnuz)

    padding_size = 0 if not padding else (16 - (block_size * 4) % 16) % 16
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4 + padding_size)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = sf.view(
        num_blocks, block_size
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4 + padding_size)


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(
                k_offsets[None, None, :] <= q_offsets[None, :, None], s, float("-inf")
            )
    return logits


# def ref_fp8_paged_mqa_logits_ragged(
#     q: torch.Tensor,
#     kv_cache: torch.Tensor,
#     weights: torch.Tensor,
#     prefix_sum_context_lens: torch.Tensor,
#     kv_indices: torch.Tensor,
#     max_model_len: int,
# ):
#     batch_size, next_n, heads, dim = q.size()
#     seq_kv, block_size, dim = kv_cache.size()  # 3d
#     logits = torch.full(
#         [batch_size * next_n, max_model_len],
#         float("-inf"),
#         device=q.device,
#         dtype=torch.float32,
#     )
#     prefix_sum_context_lens = prefix_sum_context_lens.tolist()
#     for i in range(batch_size):
#         context_len = prefix_sum_context_lens[i + 1] - prefix_sum_context_lens[i]
#         q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
#         qx, kx = (
#             q[i],
#             kv_cache[
#                 kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]]
#             ],
#         )
#         k_offsets = torch.arange(0, context_len, device="cuda")
#         mask = (k_offsets[None, :] < context_len) & (
#             k_offsets[None, :] <= q_offsets[:, None]
#         )
#         s = torch.where(
#             mask[None, :, :],
#             (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
#             float("-inf"),
#         )
#         weight_slice = (
#             weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
#         )
#         s = torch.relu(s) * weight_slice[..., None]
#         s = s.sum(dim=0)
#         logits[i * next_n : (i + 1) * next_n, :context_len] = torch.where(
#             k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
#         )

#     return logits


# create_paged_mqa_logits_configs removed — replaced by a simple loop in run_benchmark


def cosine_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denominator)


def eval_accuracy(out: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, label: str, doCheckAllClose: bool = False, text: str ="") -> float:
    out_m = out.masked_fill(~mask, 0)
    ref_m = ref.masked_fill(~mask, 0)
    diff = cosine_diff(out_m, ref_m)
    print(f"  {label} cosine_diff = {diff:.6f} {text}")
    if doCheckAllClose:
        checkAllclose(out_m, ref_m, atol=ATOL, rtol=RTOL, msg=f"{label} vs ref: ")
    return diff


def make_inputs(
    batch_size: int, next_n: int, heads: int, index_dim: int,
    avg_kv_length: int, max_model_len: int,
    blocksize: int = 1, var_ratio: float = 0.005, padding: bool = False,
    seed: int = 0,
):
    """Create all kernel inputs plus reference logits and causal mask."""
    torch.manual_seed(seed)
    random.seed(seed)

    num_blocks = cdiv(max_model_len, blocksize)
    # lo = max(1, int((1 - var_ratio) * avg_kv_length))
    lo = avg_kv_length
    hi = int((1 + var_ratio) * avg_kv_length) + 1
    context_lens = torch.randint(lo, hi, (batch_size,), device="cuda", dtype=torch.int32)

    q        = torch.randn((batch_size, next_n, heads, index_dim), device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, blocksize, 1, index_dim),  device="cuda", dtype=torch.bfloat16)
    weights  = torch.randn((batch_size * next_n, heads),           device="cuda", dtype=torch.float32)

    max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
    block_tables  = torch.zeros((batch_size, max_block_len), device="cuda", dtype=torch.int32)
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    counter = 0
    for i in range(batch_size):
        for j in range(cdiv(int(context_lens[i].item()), blocksize)):
            block_tables[i][j] = block_idx_pool[counter % num_blocks]
            counter += 1

    q_fp8        = q.to(torch.float8_e4m3fnuz)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, padding=padding)

    ref_logits = ref_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len)

    positions     = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(batch_size * next_n, -1)
    row_indices   = torch.arange(batch_size * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
    mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

    return q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask


def tune_moreh(args: argparse.Namespace):
    """Grid-search over num_warps × ChunkK × SplitKV.

    For each config: checks accuracy against the reference, then measures latency.
    Returns the fastest correct (num_warps, ChunkK, SplitKV).
    """
    blocksize     = args.blocksize if args.kv_preshuffle else 1
    batch_size    = args.batch
    next_n        = args.mtp + 1
    heads         = args.heads
    index_dim     = args.index_dim
    avg_kv_length = args.kv_length
    max_model_len = 202752

    q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask = make_inputs(
        batch_size, next_n, heads, index_dim, avg_kv_length, max_model_len,
        blocksize=blocksize, var_ratio=0.002, padding=args.padding,
    )

    print(f"[tune] B={batch_size} next_n={next_n} heads={heads} index_dim={index_dim}"
          f" avg_kv_length={avg_kv_length} max_model_len={max_model_len}")
    
    num_warps_list = [2, 4, 6, 8]
    chunk_k_list   = [64, 96, 128, 256]
    split_kv_list  = [-1] + list(range(1, 200, 1))   # -1 = auto formula inside the kernel

    results = []
    for num_warps in num_warps_list:
        for chunk_k in chunk_k_list:
            for split_kv in split_kv_list:
                try:
                    out, elapsed_us = run_perftest(
                        moreh_fp8_paged_mqa_logits,
                        q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
                        ChunkK=chunk_k, SplitKV=split_kv, num_warps=num_warps,
                        TotalCuCount=get_num_compute_units(),
                    )
                    diff = eval_accuracy(out, ref_logits, mask,
                                        f"nw={num_warps} ck={chunk_k:3d} skv={split_kv:3d}", text=f" -> {elapsed_us=} us")
                    results.append((elapsed_us, num_warps, chunk_k, split_kv, diff))
                except Exception as exc:
                    print(f"  tune: num_warps={num_warps}, ChunkK={chunk_k:3d}, SplitKV={split_kv:3d}"
                          f"  -> FAILED ({exc})")

    if not results:
        raise RuntimeError("All tune configs failed")

    results.sort()
    print("\n--- Top-5 configs ---")
    for elapsed, nw, ck, skv, diff in results[:5]:
        print(f"  num_warps={nw}, ChunkK={ck:3d}, SplitKV={skv:3d}"
              f"  -> {elapsed:.2f} us  cosine_diff={diff:.6f}")

    best_elapsed, best_nw, best_ck, best_skv, _ = results[0]
    print(f"\n>>> Best: num_warps={best_nw}, ChunkK={best_ck}, SplitKV={best_skv}"
          f"  ({best_elapsed:.2f} us)")
    return best_nw, best_ck, best_skv


def run_benchmark(args: argparse.Namespace):
    # Deepgemm baseline — matches E2E config in rocm_aiter_mla_sparse.py
    MOREH_CHUNK_K   = 128
    MOREH_SPLIT_KV  = 112
    MOREH_NUM_WARPS = 4

    blocksize = args.blocksize if args.kv_preshuffle else 1
    assert blocksize == 1 or (args.kv_preshuffle and blocksize % 16 == 0)

    if args.perf:
        shape_list = [
            # (1, args.mtp + 1, args.heads, args.index_dim, args.kv_length),
            # (2, args.mtp + 1, args.heads, args.index_dim, args.kv_length),
            # (4, args.mtp + 1, args.heads, args.index_dim, args.kv_length),
            (8, args.mtp + 1, args.heads, args.index_dim, args.kv_length),
        ]
    else:
        shape_list = [(args.batch, args.mtp + 1, args.heads, args.index_dim, args.kv_length)]

    # Realistic var_ratio values for ISL~70k / OSL~300 workloads:
    #   0.01 → context_lens within ±1% of avg  (most realistic for this workload)
    #   0.05 → ±5%  (moderate spread)
    #   0.10 → ±10% (stress-test load imbalance)
    var_ratio_list = [0.002]

    rows = []
    for (batch_size, next_n, heads, index_dim, avg_kv_length) in shape_list:
        max_model_len = 202752

        for var_ratio in var_ratio_list:
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask = make_inputs(
                batch_size, next_n, heads, index_dim, avg_kv_length, max_model_len,
                blocksize=blocksize, var_ratio=var_ratio, padding=args.padding,
            )

            # ---------------------------------------------------------- deepgemm (E2E config)
            out_logits = torch.full(
                (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
            )
            _, deepgemm_us = run_perftest(
                deepgemm_fp8_paged_mqa_logits,
                q_fp8, kv_cache_fp8, weights, out_logits, context_lens, block_tables, max_model_len,
                ChunkK=256, Preshuffle=blocksize % 16 == 0, KVBlockSize=blocksize,
                TotalCuCount=get_num_compute_units(),
            )

            # ---------------------------------------------------------- moreh kernel
            moreh_out, moreh_us = run_perftest(
                moreh_fp8_paged_mqa_logits,
                q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
                ChunkK=MOREH_CHUNK_K, SplitKV=MOREH_SPLIT_KV, num_warps=MOREH_NUM_WARPS,
                TotalCuCount=get_num_compute_units()
            )

            print(f"\n[B={batch_size} next_n={next_n} var_ratio={var_ratio}]")
            eval_accuracy(out_logits, ref_logits, mask, "deepgemm vs ref")
            eval_accuracy(moreh_out,  ref_logits, mask, "moreh    vs ref")
            eval_accuracy(moreh_out,  out_logits, mask, "moreh    vs deepgemm")
            print(f"  deepgemm : {deepgemm_us:.2f} us")
            print(f"  moreh    : {moreh_us:.2f} us")
            print(f"  speedup  : {deepgemm_us / moreh_us:.2f}x")

            rows.append({
                "batch_size"   : batch_size,
                "next_n"       : next_n,
                "heads"        : heads,
                "index_dim"    : index_dim,
                "avg_kv_length": avg_kv_length,
                "var_ratio"    : var_ratio,
                "deepgemm_us"  : round(deepgemm_us, 1),
                "moreh_us"     : round(moreh_us, 1),
                "speedup"      : f"{deepgemm_us / moreh_us:.2f}x",
            })

    df = pd.DataFrame(rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length"):
        df[col] = df[col].astype(int)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("-hq", "--heads", type=int, default=32, help="Number of query heads.")
    parser.add_argument("--index_dim", type=int, default=128, help="Head dimension.")
    parser.add_argument("-kv_length", type=int, default=70000, help="Average KV sequence length.")
    parser.add_argument("-mtp", type=int, default=0, help="Q sequence length (mtp+1 == qo_len) in MTP mode.")
    parser.add_argument("-p", "--padding", action="store_true", help="Pad KVCache contiguous dim to multiple of 16 B.")
    parser.add_argument("--perf", action="store_true", help="Sweep batch sizes 1/2/4/8.")
    parser.add_argument("--tune", action="store_true", help="Grid-search num_warps × ChunkK × SplitKV.")
    parser.add_argument("--kv_preshuffle", action="store_true", help="Enable KV cache preshuffle (blocksize must be multiple of 16).")
    parser.add_argument("--blocksize", type=int, default=1, help="KVCache block size (only with --kv_preshuffle).")
    args = parser.parse_args()

    if args.tune:
        tune_moreh(args)
    else:
        run_benchmark(args)
