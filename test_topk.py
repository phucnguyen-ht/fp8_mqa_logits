import torch
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits_stage1, deepgemm_fp8_paged_mqa_logits
# import aiter.ops.triton.attention.pa_mqa_logits as _pamqa
# # Force the non-gluon (regular triton) path. The gluon kernel uses gl.amd.AMDMFMALayout
# # which on triton 3.6 requires instr_shape in (M, N, K) format — incompatible with the
# # kernel sources here, so disable gluon entirely.
# _pamqa.enable_gluon_pa_mqa_logits = False
# _pamqa.enable_jit_gluon_pa_mqa_logits_kernel = False

def torch_ref(q, kv_cache, weights, context_lens, block_tables, max_model_len):
    """Bf16 reference. Matches `fp8_paged_mqa_logits_torch` in
    vllm/v1/attention/ops/rocm_aiter_mla_sparse.py."""
    bs, next_n, heads, hd = q.size()
    fp8 = torch.float8_e4m3fn
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


def ref_fp8_paged_mqa_logits2(
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

import functools
from moreh_fp8_paged_mqa_logits import fp8_paged_mqa_logits as moreh_fp8_paged_mqa_logits

@functools.cache
def get_num_compute_units() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count

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
    q = torch.randn(bs, next_n, heads, hd, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(bs * next_n, heads, device="cuda")
    context_lens = torch.full((bs,), ctx_len, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(
        max_blocks, dtype=torch.int32, device="cuda",
    ).unsqueeze(0).expand(bs, -1).contiguous()

    out_logits = torch.full(
        (heads, bs * next_n, max_model_len),
        float("-inf"), device="cuda", dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits_stage1(
        q, kv_cache, weights, out, context_lens, block_tables, max_model_len,
        ChunkQ=heads,
        ChunkK=256,
    )

    # out_logits = torch.full(
    #     (bs * next_n, max_model_len),
    #     float("-inf"),
    #     device="cuda",
    #     dtype=torch.float32,
    # )
    # deepgemm_fp8_paged_mqa_logits(
    #     q,
    #     kv_cache,
    #     weights,
    #     out_logits,
    #     context_lens,
    #     block_tables,
    #     max_model_len,
    #     ChunkK=256,
    #     TotalCuCount=get_num_compute_units(),
    # )

    DEFAULT_CHUNK_K   = 64
    DEFAULT_SPLIT_KV  = 96
    DEFAULT_NUM_WARPS = 4
    moreh_v2_out = moreh_fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, max_model_len,
        ChunkK=DEFAULT_CHUNK_K, SplitKV=DEFAULT_SPLIT_KV, num_warps=DEFAULT_NUM_WARPS,
        TotalCuCount=get_num_compute_units(), version=2,
    )

    moreh_v3_out = moreh_fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, max_model_len,
        ChunkK=DEFAULT_CHUNK_K, SplitKV=DEFAULT_SPLIT_KV, num_warps=DEFAULT_NUM_WARPS,
        TotalCuCount=get_num_compute_units(), version=3,
    )
    
    a = out_logits.sum(dim=0)
    r = ref_fp8_paged_mqa_logits2(q, kv_cache, weights, context_lens, block_tables, max_model_len)

    a2 = torch.where(torch.isfinite(a), a, torch.full_like(a, -1e30))
    r2 = torch.where(torch.isfinite(r), r, torch.full_like(r, -1e30))
    m2 = torch.where(torch.isfinite(moreh_v2_out), moreh_v2_out, torch.full_like(r, -1e30))
    m3 = torch.where(torch.isfinite(moreh_v3_out), moreh_v3_out, torch.full_like(r, -1e30))

    k = min(2048, ctx_len)
    a_set = torch.topk(a2, k=k, dim=1).indices.sort(dim=1).values
    r_set = torch.topk(r2, k=k, dim=1).indices.sort(dim=1).values
    m2_set = torch.topk(m2, k=k, dim=1).indices.sort(dim=1).values
    m3_set = torch.topk(m3, k=k, dim=1).indices.sort(dim=1).values
    match_ar = (a_set == r_set).float().mean(dim=1)
    match_am2 = (a_set == m2_set).float().mean(dim=1)
    match_am3 = (a_set == m3_set).float().mean(dim=1)
    match_rm2 = (r_set == m2_set).float().mean(dim=1)
    match_rm3 = (r_set == m3_set).float().mean(dim=1)

    print(f"ctx={ctx_len:5d}  topk_set_match_ar={match_ar.mean().item():.4f}")
    print(f"ctx={ctx_len:5d}  topk_set_match_am2={match_am2.mean().item():.4f}")
    print(f"ctx={ctx_len:5d}  topk_set_match_am3={match_am3.mean().item():.4f}")
    print(f"ctx={ctx_len:5d}  topk_set_match_rm2={match_rm2.mean().item():.4f}")
    print(f"ctx={ctx_len:5d}  topk_set_match_rm3={match_rm3.mean().item():.4f}")


for ctx in [1024, 2048, 2049, 3000, 4096, 8192, 16384, 32768, 65536]:
    test(ctx)