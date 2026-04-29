# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import argparse
import functools

import torch
import os
import pandas as pd

from aiter.test_common import run_perftest, checkAllclose
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits, deepgemm_fp8_paged_mqa_logits_stage1
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

def _deepgemm_fp8_paged_mqa_logits_stage1(
    q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
    heads, batch_size, next_n, out_qk, 
    ChunkK, TotalCuCount,
):
    deepgemm_fp8_paged_mqa_logits_stage1(
        q_fp8,
        kv_cache_fp8,
        weights,
        out_qk,
        context_lens,
        block_tables,
        max_model_len,
        ChunkQ=heads,
        ChunkK=ChunkK,
        TotalCuCount=TotalCuCount
    )
    return out_qk.sum(dim=0)

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


def parse_batch_arg(s: str) -> list:
    """Parse --batch: '8' | '1-20' | '1,2,4,8'."""
    s = s.strip()
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def tune_moreh(args: argparse.Namespace):
    """Grid-search num_warps × ChunkK × SplitKV across a list of batch_sizes.

    For each batch_size: re-inits inputs, runs deepgemm baseline, grid-searches
    moreh configs, keeps the top-10, and writes a CSV row.
    """
    blocksize     = args.blocksize if args.kv_preshuffle else 1
    batch_sizes   = parse_batch_arg(args.batch)
    next_n        = args.mtp + 1
    heads         = args.heads
    index_dim     = args.index_dim
    avg_kv_length = args.kv_length
    max_model_len = 202752
    var_ratio     = 0.002

    STEP_TUNE      = 4
    num_warps_list = [2, 4, 6, 8]
    chunk_k_list   = [64, 96, 128, 256]
    split_kv_list  = [-1] + list(range(STEP_TUNE, 400, STEP_TUNE))

    ACC_DIFF_THRESHOLD = 0.1
    warn_path = os.path.splitext(args.tune_csv)[0] + "_warnings.log"
    warn_f = open(warn_path, "w")
    warn_f.write(
        f"# Accuracy warnings: configs with cosine_diff > {ACC_DIFF_THRESHOLD}\n"
        f"# shared inputs: next_n={next_n} heads={heads} index_dim={index_dim}"
        f" avg_kv_length={avg_kv_length} max_model_len={max_model_len}"
        f" blocksize={blocksize} var_ratio={var_ratio}\n"
    )
    warn_f.flush()

    rows = []
    top1_rows = []
    for batch_size in batch_sizes:
        print(f"\n[tune] B={batch_size} next_n={next_n} heads={heads} index_dim={index_dim}"
              f" avg_kv_length={avg_kv_length} max_model_len={max_model_len}")

        q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask = make_inputs(
            batch_size, next_n, heads, index_dim, avg_kv_length, max_model_len,
            blocksize=blocksize, var_ratio=var_ratio, padding=args.padding,
        )

        out_logits = torch.full(
            (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
        )
        _, deepgemm_us = run_perftest(
            deepgemm_fp8_paged_mqa_logits,
            q_fp8, kv_cache_fp8, weights, out_logits, context_lens, block_tables, max_model_len,
            ChunkK=256, Preshuffle=blocksize % 16 == 0, KVBlockSize=blocksize,
            TotalCuCount=get_num_compute_units(),
        )

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
                                            f"B={batch_size} nw={num_warps} ck={chunk_k:3d} skv={split_kv:3d}",
                                            text=f" -> {elapsed_us=} us")
                        if diff > ACC_DIFF_THRESHOLD:
                            warn_f.write(
                                f"B={batch_size} num_warps={num_warps} ChunkK={chunk_k} SplitKV={split_kv}"
                                f" -> cosine_diff={diff:.6f}\n"
                            )
                            warn_f.flush()
                        results.append((elapsed_us, num_warps, chunk_k, split_kv, diff))
                    except Exception as exc:
                        print(f"  tune: B={batch_size} num_warps={num_warps}, ChunkK={chunk_k:3d}, SplitKV={split_kv:3d}"
                              f"  -> FAILED ({exc})")

        if not results:
            print(f"  All tune configs failed for B={batch_size}, skipping")
            continue

        results.sort()
        top10 = results[:10]
        print(f"\n--- Top-10 configs (B={batch_size}) ---")
        for elapsed, nw, ck, skv, diff in top10:
            print(f"  num_warps={nw}, ChunkK={ck:3d}, SplitKV={skv:3d}"
                  f"  -> {elapsed:.2f} us  cosine_diff={diff:.6f}")

        best_us       = top10[0][0]
        best_nw       = top10[0][1]
        best_ck       = top10[0][2]
        best_skv      = top10[0][3]
        top_configs   = [(nw, ck, skv, int(elapsed)) for elapsed, nw, ck, skv, _ in top10]

        rows.append({
            "batch_size"   : batch_size,
            "next_n"       : next_n,
            "heads"        : heads,
            "index_dim"    : index_dim,
            "avg_kv_length": avg_kv_length,
            "var_ratio"    : var_ratio,
            "deepgemm_us"  : round(deepgemm_us, 1),
            "moreh_us"     : round(best_us, 1),
            "top_configs"  : str(top_configs),
        })

        top1_rows.append({
            "batch_size"   : batch_size,
            "next_n"       : next_n,
            "heads"        : heads,
            "index_dim"    : index_dim,
            "avg_kv_length": avg_kv_length,
            "var_ratio"    : var_ratio,
            "deepgemm_us"  : round(deepgemm_us, 1),
            "moreh_us"     : round(best_us, 1),
            "num_warp"     : best_nw,
            "chunkK"       : best_ck,
            "splitK"       : best_skv,
        })

    warn_f.close()
    print(f"Saved accuracy warnings to {warn_path}")

    if not rows:
        raise RuntimeError("All tune configs failed across all batch_sizes")

    df = pd.DataFrame(rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length"):
        df[col] = df[col].astype(int)

    csv_path = args.tune_csv
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tune CSV to {csv_path}")
    print(df.to_string(index=False))

    df_top1 = pd.DataFrame(top1_rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length",
                "num_warp", "chunkK", "splitK"):
        df_top1[col] = df_top1[col].astype(int)
    top1_path = os.path.splitext(args.tune_csv)[0] + "_top1.csv"
    df_top1.to_csv(top1_path, index=False)
    print(f"\nSaved top-1 tune CSV to {top1_path}")
    print(df_top1.to_string(index=False))


def _load_tuned_configs(csv_path: str) -> dict:
    """Return {(batch_size, next_n, heads, index_dim): (num_warp, chunkK, splitK)}."""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    table = {}
    for _, r in df.iterrows():
        key = (int(r["batch_size"]), int(r["next_n"]),
               int(r["heads"]),      int(r["index_dim"]))
        table[key] = (int(r["num_warp"]), int(r["chunkK"]), int(r["splitK"]))
    return table


def run_benchmark(args: argparse.Namespace):
    # Default moreh config — used when no tuned entry matches.
    DEFAULT_CHUNK_K   = 64
    DEFAULT_SPLIT_KV  = 120
    DEFAULT_NUM_WARPS = 8

    tuned_configs = _load_tuned_configs(args.tuned_csv) if args.use_tuned else {}
    if args.use_tuned:
        print(f"[run_benchmark] Loaded {len(tuned_configs)} tuned configs from {args.tuned_csv}")

    blocksize = args.blocksize if args.kv_preshuffle else 1
    assert blocksize == 1 or (args.kv_preshuffle and blocksize % 16 == 0)

    shape_list = [
        (bs, args.mtp + 1, args.heads, args.index_dim, args.kv_length)
        for bs in parse_batch_arg(args.batch)
    ]

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

            deepgemm_stage1_out = torch.full(
                (heads, batch_size * next_n, max_model_len),
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            )
            deepgemm_stage1_out, deepgemm_stage1_us = run_perftest(
                _deepgemm_fp8_paged_mqa_logits_stage1,
                q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
                heads, batch_size, next_n, deepgemm_stage1_out, 
                ChunkK=256, TotalCuCount=get_num_compute_units(),
            )

            # ---------------------------------------------------------- moreh kernel
            tuned = tuned_configs.get((batch_size, next_n, heads, index_dim))
            if tuned is not None:
                m_nw, m_ck, m_skv = tuned
                print(f"  [tuned] B={batch_size}: num_warps={m_nw} ChunkK={m_ck} SplitKV={m_skv}")
            else:
                m_nw, m_ck, m_skv = DEFAULT_NUM_WARPS, DEFAULT_CHUNK_K, DEFAULT_SPLIT_KV
            moreh_out, moreh_us = run_perftest(
                moreh_fp8_paged_mqa_logits,
                q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
                ChunkK=m_ck, SplitKV=m_skv, num_warps=m_nw,
                TotalCuCount=get_num_compute_units()
            )

            print(f"\n[B={batch_size} next_n={next_n} var_ratio={var_ratio}]")
            eval_accuracy(out_logits,          ref_logits, mask, "deepgemm        vs ref", doCheckAllClose=True)
            eval_accuracy(deepgemm_stage1_out, ref_logits, mask, "deepgemm_stage1 vs ref", doCheckAllClose=True)
            eval_accuracy(moreh_out,           ref_logits, mask, "moreh           vs ref: ", doCheckAllClose=True)
            eval_accuracy(moreh_out,           out_logits, mask, "moreh           vs deepgemm: ", doCheckAllClose=True)
            eval_accuracy(deepgemm_stage1_out, moreh_out,  mask, "deepgemm_stage1 vs moreh: ", doCheckAllClose=True)
            print(f"  deepgemm        : {deepgemm_us:.2f} us")
            print(f"  deepgemm_stage1 : {deepgemm_stage1_us:.2f} us")
            print(f"  moreh           : {moreh_us:.2f} us")
            print(f"  speedup (deepgemm        / moreh) : {deepgemm_us / moreh_us:.2f}x")
            print(f"  speedup (deepgemm_stage1 / moreh) : {deepgemm_stage1_us / moreh_us:.2f}x")

            rows.append({
                "batch_size"         : batch_size,
                "next_n"             : next_n,
                "heads"              : heads,
                "index_dim"          : index_dim,
                "avg_kv_length"      : avg_kv_length,
                "var_ratio"          : var_ratio,
                "deepgemm_us"        : round(deepgemm_us, 1),
                "deepgemm_stage1_us" : round(deepgemm_stage1_us, 1),
                "moreh_us"           : round(moreh_us, 1),
                "speedup_deepgemm"   : f"{deepgemm_us / moreh_us:.2f}x",
                "speedup_stage1"     : f"{deepgemm_stage1_us / moreh_us:.2f}x",
            })

    df = pd.DataFrame(rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length"):
        df[col] = df[col].astype(int)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=str, default="8",
                        help="Batch size(s). Accepts '8', '1-20', or '1,2,4,8'.")
    parser.add_argument("-hq", "--heads", type=int, default=32, help="Number of query heads.")
    parser.add_argument("--index_dim", type=int, default=128, help="Head dimension.")
    parser.add_argument("-kv_length", type=int, default=65536, help="Average KV sequence length.")
    parser.add_argument("-mtp", type=int, default=0, help="Q sequence length (mtp+1 == qo_len) in MTP mode.")
    parser.add_argument("-p", "--padding", action="store_true", help="Pad KVCache contiguous dim to multiple of 16 B.")
    parser.add_argument("--tune", action="store_true", help="Grid-search num_warps × ChunkK × SplitKV.")
    parser.add_argument("--tune_csv", type=str, default="tune_results.csv",
                        help="CSV output path for --tune mode.")
    parser.add_argument("--kv_preshuffle", action="store_true", help="Enable KV cache preshuffle (blocksize must be multiple of 16).")
    parser.add_argument("--blocksize", type=int, default=1, help="KVCache block size (only with --kv_preshuffle).")
    parser.add_argument("--use_tuned", action="store_true",
                        help="Use tuned (num_warps, ChunkK, SplitKV) per shape from --tuned_csv.")
    parser.add_argument("--tuned_csv", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "paged_fp8_mqa_logits_tuned.csv"),
                        help="CSV with tuned (num_warp, chunkK, splitK) per shape.")
    args = parser.parse_args()

    if args.tune:
        tune_moreh(args)
    else:
        run_benchmark(args)
