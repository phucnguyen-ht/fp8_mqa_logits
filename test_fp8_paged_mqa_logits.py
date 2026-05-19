# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import argparse
import functools
import gc
import time

import torch
import os
import pandas as pd

from aiter_testcommon import run_perftest, checkAllclose
# import aiter.ops.triton.attention.pa_mqa_logits as _pamqa
# Force the non-gluon (regular triton) path. The gluon kernel uses gl.amd.AMDMFMALayout
# which on triton 3.6 requires instr_shape in (M, N, K) format — incompatible with the
# kernel sources here, so disable gluon entirely.
# _pamqa.enable_gluon_pa_mqa_logits = False
# _pamqa.enable_jit_gluon_pa_mqa_logits_kernel = False
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


def eval_accuracy(out: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, label: str, doCheckAllClose: bool = False, text: str ="", log_file=None) -> float:
    out_m = out.masked_fill(~mask, 0)
    ref_m = ref.masked_fill(~mask, 0)
    diff = cosine_diff(out_m, ref_m)
    msg = f"  {label} cosine_diff = {diff:.6f} {text}"
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()
    else:
        print(msg)
    if doCheckAllClose:
        checkAllclose(out_m, ref_m, atol=ATOL, rtol=RTOL, msg=f"{label}: ")
    return diff


def make_inputs(
    batch_size: int, next_n: int, heads: int, index_dim: int,
    avg_kv_length: int, max_model_len: int,
    blocksize: int = 1, var_ratio: float = 0.005, padding: bool = False,
    seed: int = 0, reference = True
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

    if reference:
        ref_logits = ref_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len)
    else:
        ref_logits = None

    positions     = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(batch_size * next_n, -1)
    row_indices   = torch.arange(batch_size * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
    mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

    return q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask


def parse_batch_arg(s: str) -> list:
    """Parse --batch: '8' | '1-20' | '1,2,4,8' | '1-32,64,128,256'."""
    s = s.strip()
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def quick_time(func, *args, num_iters: int = 50, num_warmup: int = 5, **kwargs):
    """Lightweight timer for tuning — no torch.profiler / ROCTracer / deepcopy.

    Uses CUDA events. Returns (last_output, avg_us_per_iter). The aiter run_perftest
    triggers a steady ~250 MB/call leak (likely torch.profiler + ROCTracer state on
    ROCm), which OOMs the GPU within ~550 configs. This avoids that path entirely.
    """
    for _ in range(num_warmup):
        out = func(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        out = func(*args, **kwargs)
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000.0 / num_iters  # ms -> us, average per iter
    return out, avg_us

def run_perftest_identity(func, *args, num_iters: int = 50, num_warmup: int = 5, **kwargs):
    """Lightweight timer for tuning — no torch.profiler / ROCTracer / deepcopy.

    Uses CUDA events. Returns (last_output, avg_us_per_iter). The aiter run_perftest
    triggers a steady ~250 MB/call leak (likely torch.profiler + ROCTracer state on
    ROCm), which OOMs the GPU within ~550 configs. This avoids that path entirely.
    """
    out = func(*args, **kwargs)
    return out, 0

def make_split_kv_list(batch_size: int, next_n: int, avg_kv_length: int,
                       min_chunk_k: int, step: int = 8, hard_cap: int = 400) -> list:
    """Pick a SplitKV grid sized to the workload.

    For tiny ctx_len (e.g. 512) the kernel needs only ceil(ctx/min_ck) "real" splits;
    extra ones become no-op blocks. Bound the upper end by max(real_splits, gpu_fill)
    with some headroom, so we don't waste time tuning over noise-only territory.
    """
    real_splits = cdiv(avg_kv_length, min_chunk_k)
    tiles       = max(1, batch_size * next_n)
    gpu_fill    = cdiv(get_num_compute_units(), tiles) * 2  # 2x for occupancy headroom
    upper       = min(hard_cap, max(real_splits, gpu_fill) * 2)
    upper       = max(upper, step)  # always have at least one tunable point
    return [-1, 2, 4, 8] + list(range(step, upper + 1, step))


def tune_moreh(args: argparse.Namespace):
    """Grid-search num_warps × ChunkK × SplitKV across a list of batch_sizes.

    For each batch_size: re-inits inputs, runs deepgemm baseline, grid-searches
    moreh configs, keeps the top-10, and writes a CSV row.
    """
    blocksize      = args.blocksize if args.kv_preshuffle else 1
    batch_sizes    = parse_batch_arg(args.batch)
    mtp_list       = parse_batch_arg(args.mtp)
    kv_length_list = parse_batch_arg(args.kv_length)
    heads          = args.heads
    index_dim      = args.index_dim
    max_model_len  = 202752

    STEP_TUNE      = 16
    num_warps_list = [2, 4, 8]
    chunk_k_list   = [64, 128, 256]
    # split_kv_list is now built per-batch_size from make_split_kv_list()

    ACC_DIFF_THRESHOLD = 0.01
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        warn_path   = os.path.join(args.log_dir, "warnings.log")
        detail_path = os.path.join(args.log_dir, "detail.log")
    else:
        warn_path   = os.path.splitext(args.tune_csv)[0] + "_warnings.log"
        detail_path = os.path.splitext(args.tune_csv)[0] + "_detail.log"

    # Append mode: when tune.sh launches one Python invocation per batch_size, multiple
    # invocations share the same log_dir (per (mtp, kv_length)) and accumulate together.
    warn_f = open(warn_path, "a")
    warn_f.write(f"# Accuracy warnings: configs with cosine_diff > {ACC_DIFF_THRESHOLD}\n")
    warn_f.flush()

    detail_f = open(detail_path, "a")
    detail_f.write("# Per-config tune log\n")
    detail_f.flush()

    # Resume support: read done_keys from --resume_csv (shared final CSV) if provided,
    # otherwise fall back to --tune_csv itself (single-GPU / standalone usage).
    csv_path  = args.tune_csv
    top1_path = os.path.splitext(csv_path)[0] + "_top1.csv"
    resume_path      = args.resume_csv if args.resume_csv else csv_path
    resume_top1_path = os.path.splitext(resume_path)[0] + "_top1.csv"
    key_cols = ["batch_size", "next_n", "heads", "index_dim", "avg_kv_length"]
    # done_keys: read from shared final CSV (resume_path) to skip already-tuned configs
    resume_full_df = pd.read_csv(resume_path)      if os.path.isfile(resume_path)      else None
    resume_top1_df = pd.read_csv(resume_top1_path) if os.path.isfile(resume_top1_path) else None
    src_for_keys = resume_top1_df if resume_top1_df is not None else resume_full_df
    if src_for_keys is not None and len(src_for_keys):
        done_keys = set(map(tuple, src_for_keys[key_cols].astype(int).values.tolist()))
    else:
        done_keys = set()
    # existing rows: read from per-GPU CSV (csv_path) to accumulate results across invocations
    existing_full_df = pd.read_csv(csv_path)      if os.path.isfile(csv_path)      else None
    existing_top1_df = pd.read_csv(top1_path) if os.path.isfile(top1_path) else None

    rows = []
    top1_rows = []
    for avg_kv_length in kv_length_list:
        var_ratio = 256 / avg_kv_length
        for mtp in mtp_list:
            next_n = mtp + 1
            for batch_size in batch_sizes:
                key = (batch_size, next_n, heads, index_dim, avg_kv_length)
                shared_header = (
                    f"# shared inputs: batch_size={batch_size} next_n={next_n}"
                    f" heads={heads} index_dim={index_dim}"
                    f" avg_kv_length={avg_kv_length} max_model_len={max_model_len}"
                    f" blocksize={blocksize} var_ratio={var_ratio}\n"
                )
                warn_f.write(shared_header)
                warn_f.flush()
                detail_f.write(shared_header)
                detail_f.flush()
                if key in done_keys:
                    print(f"[skip] B={batch_size} next_n={next_n} heads={heads}"
                          f" index_dim={index_dim} avg_kv_length={avg_kv_length} already tuned")
                    continue
                
                print(f"\n[tune] B={batch_size} next_n={next_n} heads={heads} index_dim={index_dim}"
                      f" avg_kv_length={avg_kv_length} max_model_len={max_model_len}")
    
                q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask = make_inputs(
                    batch_size, next_n, heads, index_dim, avg_kv_length, max_model_len,
                    blocksize=blocksize, var_ratio=var_ratio, padding=args.padding,
                    seed=int(time.time_ns()) & 0x7FFFFFFF,
                )
    
                # Max ctx_len after var_ratio variation. SplitKV is derived from this per chunk_k.
                max_ctx_len = int(avg_kv_length * (1 + var_ratio))
    
                out_logits = torch.full(
                    (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
                )
                try:
                    _, deepgemm_us = run_perftest(
                        deepgemm_fp8_paged_mqa_logits,
                        q_fp8, kv_cache_fp8, weights, out_logits, context_lens, block_tables, max_model_len,
                        ChunkK=256, Preshuffle=blocksize % 16 == 0, KVBlockSize=blocksize,
                        TotalCuCount=get_num_compute_units(),
                    )
    
                    results = []
                    for num_warps in num_warps_list:
                        for chunk_k in chunk_k_list:
                            # SplitKV search space: powers-of-2 up to 16, then multiples of 8,
                            # always including X = num_chunks (one chunk per block = max SKV).
                            num_chunks = cdiv(max_ctx_len, chunk_k)
                            skv_set = set()
                            p = 1
                            while p <= num_chunks:
                                skv_set.add(p)
                                if p >= 16:
                                    break
                                p *= 2
                            for v in range(24, num_chunks, 8):
                                skv_set.add(v)
                            skv_set.add(num_chunks)
                            split_kv_list = [-1] + sorted(skv_set)
                            # split_kv_list = list(set([-1] + list(range(1, 33)) + list(range(32, 400, 16))))
                            print(f"  ck={chunk_k:3d} -> {len(split_kv_list)} SplitKV values: {split_kv_list}")
    
                            for split_kv in split_kv_list:
                                out = None
                                try:
                                    moreh_out_logits = torch.full(
                                        (batch_size * next_n, max_model_len), float("-inf"),
                                        device="cuda", dtype=torch.float32,
                                    )
                                    out, elapsed_us = run_perftest(
                                        moreh_fp8_paged_mqa_logits,
                                        q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
                                        ChunkK=chunk_k, SplitKV=split_kv, num_warps=num_warps,
                                        TotalCuCount=get_num_compute_units(),
                                        version=args.version,
                                        out_logits=moreh_out_logits,
                                    )
                                    diff = eval_accuracy(out, ref_logits, mask,
                                                        f"B={batch_size} nw={num_warps} ck={chunk_k:3d} skv={split_kv:3d}(moreh vs ref)     ",
                                                        text=f" -> {elapsed_us=} us",
                                                        log_file=detail_f)
                                    diff_dg = eval_accuracy(out, out_logits, mask,
                                                  f"B={batch_size} nw={num_warps} ck={chunk_k:3d} skv={split_kv:3d} (moreh vs deepgemm)",
                                                  log_file=detail_f)
                                    if diff > ACC_DIFF_THRESHOLD:
                                        warn_f.write(
                                            f"B={batch_size} num_warps={num_warps} ChunkK={chunk_k} SplitKV={split_kv}"
                                            f" -> cosine_diff={diff:.6f}\n"
                                        )
                                        warn_f.flush()
                                    results.append((elapsed_us, num_warps, chunk_k, split_kv, diff, diff_dg))
                                except Exception as exc:
                                    detail_f.write(
                                        f"  tune: B={batch_size} num_warps={num_warps}, ChunkK={chunk_k:3d}, SplitKV={split_kv:3d}"
                                        f"  -> FAILED ({exc})\n"
                                    )
                                    detail_f.flush()
                                finally:
                                    # Free the per-config moreh output so PyTorch's caching allocator
                                    # can reuse the memory across the ~800 configs in the inner loop.
                                    if out is not None:
                                        del out
                                    gc.collect()
                                    torch.cuda.empty_cache()
    
                    if not results:
                        print(f"  All tune configs failed for B={batch_size}, skipping")
                        continue
    
                    results.sort()
                    top10 = results[:10]
                    print(f"\n--- Top-10 configs (B={batch_size}) ---")
                    for elapsed, nw, ck, skv, diff, diff_dg in top10:
                        print(f"  num_warps={nw}, ChunkK={ck:3d}, SplitKV={skv:3d}"
                              f"  -> {elapsed:.2f} us  cosine_diff(vs ref)={diff:.6f}"
                              f"  cosine_diff(vs deepgemm)={diff_dg:.6f}")
    
                    best_us       = top10[0][0]
                    best_nw       = top10[0][1]
                    best_ck       = top10[0][2]
                    best_skv      = top10[0][3]
                    top_configs   = [(nw, ck, skv, int(elapsed)) for elapsed, nw, ck, skv, _, _ in top10]
    
                    speedup = round(deepgemm_us / best_us, 3) if best_us > 0 else None

                    rows.append({
                        "batch_size"   : batch_size,
                        "next_n"       : next_n,
                        "heads"        : heads,
                        "index_dim"    : index_dim,
                        "avg_kv_length": avg_kv_length,
                        "var_ratio"    : var_ratio,
                        "deepgemm_us"  : round(deepgemm_us, 1),
                        "moreh_us"     : round(best_us, 1),
                        "speedup"      : speedup,
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
                        "speedup"      : speedup,
                        "num_warp"     : best_nw,
                        "chunkK"       : best_ck,
                        "splitK"       : best_skv,
                    })
                finally:
                    # Release per-batch input tensors before moving to the next batch_size.
                    # Without this, the caching allocator pins memory across batches and OOMs
                    # when the next batch tries to allocate fresh inputs.
                    del q_fp8, kv_cache_fp8, weights, context_lens, block_tables, ref_logits, mask, out_logits
                    gc.collect()
                    torch.cuda.empty_cache()

    warn_f.close()
    detail_f.close()
    print(f"Saved accuracy warnings to {warn_path}")
    print(f"Saved per-config detail log to {detail_path}")

    if not rows:
        print("\n[tune] No new configs were tuned (all skipped or all failed).")
        return

    df = pd.DataFrame(rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length"):
        df[col] = df[col].astype(int)
    if existing_full_df is not None:
        df = pd.concat([existing_full_df, df], ignore_index=True)
    df = df.sort_values(key_cols).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tune CSV to {csv_path}  (now {len(df)} rows total)")
    print(df.to_string(index=False))

    df_top1 = pd.DataFrame(top1_rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length",
                "num_warp", "chunkK", "splitK"):
        df_top1[col] = df_top1[col].astype(int)
    if existing_top1_df is not None:
        df_top1 = pd.concat([existing_top1_df, df_top1], ignore_index=True)
    df_top1 = df_top1.sort_values(key_cols).reset_index(drop=True)
    df_top1.to_csv(top1_path, index=False)
    print(f"\nSaved top-1 tune CSV to {top1_path}  (now {len(df_top1)} rows total)")
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

def run_profile(args: argparse.Namespace):
    # Default moreh config — used when no tuned entry matches.
    DEFAULT_CHUNK_K   = 64
    DEFAULT_SPLIT_KV  = 440
    DEFAULT_NUM_WARPS = 4

    tuned_configs = _load_tuned_configs(args.tuned_csv) if args.use_tuned else {}
    if args.use_tuned:
        print(f"[run_profile] Loaded {len(tuned_configs)} tuned configs from {args.tuned_csv}")

    blocksize = args.blocksize if args.kv_preshuffle else 1
    assert blocksize == 1 or (args.kv_preshuffle and blocksize % 16 == 0)

    shape_list = [
        (bs, mtp + 1, args.heads, args.index_dim, kv)
        for bs in parse_batch_arg(args.batch)
        for mtp in parse_batch_arg(args.mtp)
        for kv in parse_batch_arg(args.kv_length)
    ]

    # Realistic var_ratio values for ISL~70k / OSL~300 workloads:
    #   0.01 → context_lens within ±1% of avg  (most realistic for this workload)
    #   0.05 → ±5%  (moderate spread)
    #   0.10 → ±10% (stress-test load imbalance)

    rows = []
    for (batch_size, next_n, heads, index_dim, avg_kv_length) in shape_list:
        max_model_len = 202752
        var_ratio = 256 / avg_kv_length

        print(f"{batch_size=}, {next_n=}, {heads=}, {index_dim=}, {avg_kv_length=}, {var_ratio=}")
        q_fp8, kv_cache_fp8, weights, context_lens, block_tables, _, mask = make_inputs(
            batch_size, next_n, heads, index_dim, avg_kv_length, max_model_len,
            blocksize=blocksize, var_ratio=var_ratio, padding=args.padding, 
            reference=False
        )

        # ---------------------------------------------------------- deepgemm (E2E config)
        out_logits = torch.full(
            (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
        )
        _, deepgemm_us = run_perftest_identity(
            deepgemm_fp8_paged_mqa_logits,
            q_fp8, kv_cache_fp8, weights, out_logits, context_lens, block_tables, max_model_len,
            ChunkK=256, Preshuffle=blocksize % 16 == 0, KVBlockSize=blocksize,
            TotalCuCount=get_num_compute_units(),
            num_iters=100, num_warmup=5
        )

        # ---------------------------------------------------------- moreh kernel
        tuned = tuned_configs.get((batch_size, next_n, heads, index_dim))
        if tuned is not None:
            m_nw, m_ck, m_skv = tuned
            print(f"  [tuned] B={batch_size}: num_warps={m_nw} ChunkK={m_ck} SplitKV={m_skv}")
        else:
            m_nw, m_ck, m_skv = DEFAULT_NUM_WARPS, DEFAULT_CHUNK_K, DEFAULT_SPLIT_KV
        moreh_out_logits = torch.full(
            (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
        )
        moreh_out, moreh_us = run_perftest_identity(
            moreh_fp8_paged_mqa_logits,
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
            ChunkK=m_ck, SplitKV=m_skv, num_warps=m_nw,
            TotalCuCount=get_num_compute_units(), version=args.version,
            out_logits=moreh_out_logits,
            num_iters=100, num_warmup=5,
            version=7
        )

        # eval_accuracy(moreh_out,           out_logits, mask, "moreh           vs deepgemm: ", doCheckAllClose=True)

def run_benchmark(args: argparse.Namespace):
    # Default moreh config — used when no tuned entry matches.
    DEFAULT_CHUNK_K   = 64
    DEFAULT_SPLIT_KV  = -1
    DEFAULT_NUM_WARPS = 4

    tuned_configs = _load_tuned_configs(args.tuned_csv) if args.use_tuned else {}
    if args.use_tuned:
        print(f"[run_benchmark] Loaded {len(tuned_configs)} tuned configs from {args.tuned_csv}")

    blocksize = args.blocksize if args.kv_preshuffle else 1
    assert blocksize == 1 or (args.kv_preshuffle and blocksize % 16 == 0)

    shape_list = [
        (bs, mtp + 1, args.heads, args.index_dim, kv)
        for bs in parse_batch_arg(args.batch)
        for mtp in parse_batch_arg(args.mtp)
        for kv in parse_batch_arg(args.kv_length)
    ]

    # Realistic var_ratio values for ISL~70k / OSL~300 workloads:
    #   0.01 → context_lens within ±1% of avg  (most realistic for this workload)
    #   0.05 → ±5%  (moderate spread)
    #   0.10 → ±10% (stress-test load imbalance)

    rows = []
    for (batch_size, next_n, heads, index_dim, avg_kv_length) in shape_list:
        max_model_len = 202752
        avg_kv_length += 128
        var_ratio = 128 / avg_kv_length

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
        else:
            m_nw, m_ck, m_skv = DEFAULT_NUM_WARPS, DEFAULT_CHUNK_K, DEFAULT_SPLIT_KV

        moreh_v2_out_logits = torch.full(
            (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
        )
        moreh_v2_out, moreh_v2_us = run_perftest(
            moreh_fp8_paged_mqa_logits,
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
            ChunkK=m_ck, SplitKV=m_skv, num_warps=m_nw,
            TotalCuCount=get_num_compute_units(), version=2,
            out_logits=moreh_v2_out_logits,
        )

        moreh_v7_out_logits = torch.full(
            (batch_size * next_n, max_model_len), float("-inf"), device="cuda", dtype=torch.float32
        )
        moreh_v7_out, moreh_v7_us = run_perftest(
            moreh_fp8_paged_mqa_logits,
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len,
            ChunkK=m_ck, SplitKV=m_skv, num_warps=m_nw,
            TotalCuCount=get_num_compute_units(), version=7,
            out_logits=moreh_v7_out_logits,
        )

        print(f"\n[B={batch_size} next_n={next_n} var_ratio={var_ratio} kv_length={avg_kv_length}]"
              f" num_warps={m_nw} ChunkK={m_ck} SplitKV={m_skv}")
        eval_accuracy(out_logits,     ref_logits,   mask, "deepgemm        vs ref     ", doCheckAllClose=False)
        eval_accuracy(deepgemm_stage1_out, ref_logits, mask, "deepgemm_stage1 vs ref  ", doCheckAllClose=False)
        eval_accuracy(moreh_v2_out,   ref_logits,   mask, "moreh_v2        vs ref     ", doCheckAllClose=False)
        eval_accuracy(moreh_v7_out,   ref_logits,   mask, "moreh_v7        vs ref     ", doCheckAllClose=False)
        eval_accuracy(moreh_v7_out,   out_logits, mask, "moreh_v7        vs deepgemm", doCheckAllClose=True)
        eval_accuracy(moreh_v7_out,   moreh_v2_out, mask, "moreh_v7        vs moreh_v2", doCheckAllClose=True)
        print(f"  deepgemm        : {deepgemm_us:.2f} us")
        print(f"  deepgemm_stage1 : {deepgemm_stage1_us:.2f} us")
        print(f"  moreh_v2 (PAD)  : {moreh_v2_us:.2f} us")
        print(f"  moreh_v7 (swiz) : {moreh_v7_us:.2f} us")
        print(f"  speedup v2 / deepgemm        : {deepgemm_us / moreh_v2_us:.2f}x")
        print(f"  speedup v7 / deepgemm        : {deepgemm_us / moreh_v7_us:.2f}x")
        print(f"  speedup v7 / v2              : {moreh_v2_us / moreh_v7_us:.2f}x")
        moreh_out, moreh_us = moreh_v2_out, moreh_v2_us  # keep rows compatible

        rows.append({
            "batch_size"         : batch_size,
            "next_n"             : next_n,
            "heads"              : heads,
            "index_dim"          : index_dim,
            "avg_kv_length"      : avg_kv_length,
            "var_ratio"          : var_ratio,
            "deepgemm_us"        : round(deepgemm_us, 1),
            "deepgemm_stage1_us" : round(deepgemm_stage1_us, 1),
            "moreh_v2_us"        : round(moreh_v2_us, 1),
            "moreh_v7_us"        : round(moreh_v7_us, 1),
            "speedup_v2_deepgemm": f"{deepgemm_us / moreh_v2_us:.2f}x",
            "speedup_v7_deepgemm": f"{deepgemm_us / moreh_v7_us:.2f}x",
            "speedup_v7_vs_v2"   : f"{moreh_v2_us / moreh_v7_us:.2f}x",
        })

    df = pd.DataFrame(rows)
    for col in ("batch_size", "next_n", "heads", "index_dim", "avg_kv_length"):
        df[col] = df[col].astype(int)
    print("\n" + df.to_string(index=False))
    df.to_csv(args.run_csv, index=False)
    print(f"Saved benchmark CSV to {args.run_csv}")

# chunkk -> mỗi block xử lý chunkk*x
# splitkv = max_ctx_len / (chunkk * x)
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=str, default="8",
                        help="Batch size(s). Accepts '8', '1-20', or '1,2,4,8'.")
    parser.add_argument("-hq", "--heads", type=int, default=32, help="Number of query heads.")
    parser.add_argument("--index_dim", type=int, default=128, help="Head dimension.")
    parser.add_argument("-kv_length", type=str, default="65536", help="KV sequence length(s). Accepts '65536', '512-1024', or '512,1024,2048'.")
    parser.add_argument("-mtp", type=str, default="0", help="MTP value(s). Accepts '0', '1-3', or '0,1,2'.")
    parser.add_argument("-p", "--padding", action="store_true", help="Pad KVCache contiguous dim to multiple of 16 B.")
    parser.add_argument("--tune", action="store_true", help="Grid-search num_warps × ChunkK × SplitKV.")
    parser.add_argument("--profile", action="store_true", help="Grid-search num_warps × ChunkK × SplitKV.")
    parser.add_argument("--tune_csv", type=str, default="tune_results.csv",
                        help="CSV output path for --tune mode (resume-aware: existing "
                             "(batch_size, next_n, heads, index_dim, avg_kv_length) rows are skipped).")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to write per-run warnings.log and detail.log. "
                             "Default: derived from --tune_csv path.")
    parser.add_argument("--kv_preshuffle", action="store_true", help="Enable KV cache preshuffle (blocksize must be multiple of 16).")
    parser.add_argument("--blocksize", type=int, default=1, help="KVCache block size (only with --kv_preshuffle).")
    parser.add_argument("--use_tuned", action="store_true",
                        help="Use tuned (num_warps, ChunkK, SplitKV) per shape from --tuned_csv.")
    parser.add_argument("--tuned_csv", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "paged_fp8_mqa_logits_tuned.csv"),
                        help="CSV with tuned (num_warp, chunkK, splitK) per shape.")
    parser.add_argument("--run_csv", type=str, default="run.csv",
                        help="CSV output path for benchmark results.")
    parser.add_argument("--resume_csv", type=str, default=None,
                        help="CSV to read done_keys from for resume (multi-GPU: pass the shared "
                             "final merged CSV so all workers skip already-tuned configs).")
    parser.add_argument("--version", type=int, default=2, choices=[2, 3, 4, 5, 6, 7],
                        help="Kernel version for --tune: 2=PAD(v2), 3=PAD+bt-prefetch(v3), "
                             "4=swizzle(v4), 5=v3+direct-QW(v5). "
                             "run_benchmark always compares both v2 and v4.")
    args = parser.parse_args()
    if args.profile:
        run_profile(args)
    elif args.tune:
        tune_moreh(args)
    else:
        run_benchmark(args)
