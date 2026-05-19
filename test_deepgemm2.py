import torch
import functools
import random
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits_stage1, deepgemm_fp8_paged_mqa_logits
)
import aiter.ops.triton.attention.pa_mqa_logits as _pamqa
# Force the non-gluon (regular triton) path. The gluon kernel uses gl.amd.AMDMFMALayout
# which on triton 3.6 requires instr_shape in (M, N, K) format — incompatible with the
# kernel sources here, so disable gluon entirely.
_pamqa.enable_gluon_pa_mqa_logits = False
_pamqa.enable_jit_gluon_pa_mqa_logits_kernel = False

def torch_ref(q, kv_cache, weights, context_lens, block_tables, max_model_len):
    """Bf16 reference. Matches `fp8_paged_mqa_logits_torch` in
    vllm/v1/attention/ops/rocm_aiter_mla_sparse.py."""
    bs, next_n, heads, hd = q.size()
    fp8 = torch.float8_e4m3fnuz
    kv_cache_v, scale = kv_cache[..., :hd], kv_cache[..., hd:]
    scale = scale.contiguous().view(torch.float32)
    kv_cache_v = kv_cache_v.view(fp8).float() * scale
    _, block_size, _, _ = kv_cache_v.size()
    logits = torch.full(
        [bs * next_n, max_model_len], float("-inf"),
        device=q.device, dtype=torch.float32,
    )
    q_f = q.float()
    for i in range(bs):
        ctx_len = context_lens[i].item()
        q_offsets = torch.arange(ctx_len - next_n, ctx_len, device="cuda")
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for blk in range((ctx_len + block_size - 1) // block_size):
            blk_id = block_tables[i][blk]
            qx, kx = q_f[i], kv_cache_v[blk_id]
            k_offsets = torch.arange(blk * block_size, (blk + 1) * block_size, device="cuda")
            mask = (k_offsets[None, :] < ctx_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
                float("-inf"),
            )
            s = (torch.relu(s) * weight_slice[..., None]).sum(dim=0)
            logits[i * next_n:(i + 1) * next_n, blk * block_size:(blk + 1) * block_size] = \
                torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def inspect_kv_scales(kv_cache, hd=128, label=""):
    """Reinterpret the last 4 bytes per row of kv_cache as float32 and report
    distribution: NaN / +Inf / -Inf / denormal / finite normal + log-magnitude
    histogram. Useful to confirm whether random uint8 bytes really do produce
    NaN/Inf scales."""
    # kv_cache shape: (num_blocks, block_size, 1, hd+4)  dtype=uint8
    scale_bytes = kv_cache[..., hd:hd + 4].contiguous()
    scale_f32 = scale_bytes.view(torch.float32).flatten()
    n = scale_f32.numel()

    is_nan       = torch.isnan(scale_f32)
    is_posinf    = scale_f32 == float("inf")
    is_neginf    = scale_f32 == float("-inf")
    is_inf       = is_posinf | is_neginf
    is_finite    = torch.isfinite(scale_f32)
    abs_v        = scale_f32.abs()
    # Denormals are finite but < smallest normal (2^-126 ≈ 1.18e-38)
    is_denormal  = is_finite & (abs_v < 1.175494e-38) & (abs_v > 0)
    is_zero      = scale_f32 == 0
    is_normal    = is_finite & ~is_denormal & ~is_zero

    print(f"--- inspect_kv_scales{(' '+label) if label else ''} (n={n}) ---")
    print(f"  NaN       : {is_nan.sum().item():8d}  ({100*is_nan.float().mean().item():.3f}%)")
    print(f"  +Inf      : {is_posinf.sum().item():8d}  ({100*is_posinf.float().mean().item():.5f}%)")
    print(f"  -Inf      : {is_neginf.sum().item():8d}  ({100*is_neginf.float().mean().item():.5f}%)")
    print(f"  ±0        : {is_zero.sum().item():8d}")
    print(f"  denormal  : {is_denormal.sum().item():8d}  ({100*is_denormal.float().mean().item():.3f}%)")
    print(f"  normal    : {is_normal.sum().item():8d}  ({100*is_normal.float().mean().item():.3f}%)")
    print(f"    of normals: negative = {(is_normal & (scale_f32<0)).sum().item()}")
    if is_normal.any():
        normal_abs = abs_v[is_normal]
        print(f"    abs min / max     : {normal_abs.min().item():.3e}  /  {normal_abs.max().item():.3e}")
        log10 = torch.log10(normal_abs)
        # bucket by decade
        edges = torch.tensor([-40, -30, -20, -10, -1, 0, 1, 10, 20, 30, 40], device=log10.device, dtype=log10.dtype)
        for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
            cnt = ((log10 >= lo) & (log10 < hi)).sum().item()
            if cnt:
                print(f"    log10|x| ∈ [{lo:>4}, {hi:>4}): {cnt}")
    # how many KV rows have AT LEAST ONE problematic scale (NaN/Inf/huge)
    huge_thresh = 1e10
    is_bad = is_nan | is_inf | (is_finite & (abs_v > huge_thresh))
    n_rows_bad = is_bad.view(kv_cache.shape[0], -1).any(dim=1).sum().item()
    print(f"  rows with NaN/Inf/|x|>{huge_thresh:.0e}: {n_rows_bad}/{kv_cache.shape[0]}")
    print()


def test(ctx_len):
    torch.manual_seed(42)
    bs, next_n, heads, hd = 1, 1, 32, 128
    block_size = 1  # DeepseekV32IndexerCache uses block_size=1
    max_blocks = ctx_len + 16
    max_model_len = 202752  # GLM-5.1-FP8 config value
    kv_cache = torch.randint(
        0, 256, (max_blocks, block_size, 1, hd + 4),
        dtype=torch.uint8, device="cuda",
    )
    inspect_kv_scales(kv_cache, hd=hd, label=f"ctx={ctx_len}")
    q = torch.randn(bs, next_n, heads, hd, device="cuda").to(torch.float8_e4m3fnuz)
    weights = torch.randn(bs * next_n, heads, device="cuda")
    context_lens = torch.full((bs,), ctx_len, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(
        max_blocks, dtype=torch.int32, device="cuda",
    ).unsqueeze(0).expand(bs, -1).contiguous()
    out = torch.full(
        (heads, bs * next_n, max_model_len),
        float("-inf"), device="cuda", dtype=torch.float32,
    )

    deepgemm_fp8_paged_mqa_logits_stage1(
        q, kv_cache, weights, out, context_lens, block_tables, max_model_len, ChunkQ=heads
    )
    a = out.sum(dim=0)
    r = torch_ref(q, kv_cache, weights, context_lens, block_tables, max_model_len)

    a2 = torch.where(torch.isfinite(a), a, torch.full_like(a, -1e30))
    r2 = torch.where(torch.isfinite(r), r, torch.full_like(r, -1e30))
    k = min(2048, ctx_len)
    a_set = torch.topk(a2, k=k, dim=1).indices.sort(dim=1).values
    r_set = torch.topk(r2, k=k, dim=1).indices.sort(dim=1).values
    match = (a_set == r_set).float().mean(dim=1)
    print(f"ctx={ctx_len:5d}  topk_set_match={match.mean().item():.4f}")


for ctx in [1024, 2048, 2049, 3000, 4096, 8192, 16384, 32768, 65536]:
    test(ctx)