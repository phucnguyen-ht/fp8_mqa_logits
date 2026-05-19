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

def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y

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

def gen_inputs(
    batch_size: int, next_n: int, heads: int, index_dim: int,
    avg_kv_length: int, max_model_len: int,
    blocksize: int = 1, var_ratio: float = 0.005, padding: bool = False,
    seed: int = 0
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

    positions     = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(batch_size * next_n, -1)
    row_indices   = torch.arange(batch_size * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
    mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

    return q_fp8, kv_cache_fp8, weights, context_lens, block_tables, mask

@functools.cache
def get_num_compute_units() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count

def make_inputs(ctx_len):
    """Proper input construction matching production format: K is generated as
    bf16 randn, then `kv_cache_cast_to_fp8` packs (fp8_values || amax/240 scale).
    Scale is always finite & non-negative, so the relu/scale ordering difference
    no longer matters and Triton/HIP/torch_ref all agree."""
    bs, next_n, heads, hd = 1, 1, 32, 128
    block_size = 1
    max_model_len = 202752
    q, kv_cache, weights, context_lens, block_tables, _, = gen_inputs(
        batch_size=bs, next_n=next_n, heads=heads, index_dim=hd,
        avg_kv_length=ctx_len, max_model_len=max_model_len,
        blocksize=block_size, var_ratio=0.0, padding=False,
        seed=42
    )
    return q, kv_cache, weights, context_lens, block_tables, max_model_len

def test(ctx_len):
    q, kv_cache, weights, context_lens, block_tables, max_model_len = make_inputs(ctx_len)
    bs, next_n, heads, hd = q.size()
    ctx_len = int(context_lens[0].item())
    out = torch.full(
        (heads, bs * next_n, max_model_len),
        float("-inf"), device="cuda", dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits_stage1(
        q, kv_cache, weights, out, context_lens, block_tables, max_model_len,
        ChunkQ=heads, ChunkK=256, TotalCuCount=80
    )

    a = out.sum(dim=0)
    r = torch_ref(q, kv_cache, weights, context_lens, block_tables, max_model_len)

    k = min(2048, ctx_len)
    a_set = torch.topk(a, k=k, dim=1).indices.sort(dim=1).values
    r_set = torch.topk(r, k=k, dim=1).indices.sort(dim=1).values
    match = (a_set == r_set).float().mean(dim=1)
    print(f"ctx={ctx_len:5d}  topk_set_match={match.mean().item():.4f}")

for ctx in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    test(ctx)