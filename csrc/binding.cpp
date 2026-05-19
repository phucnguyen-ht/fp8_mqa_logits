#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <optional>

torch::Tensor fp8_paged_mqa_logits(
    torch::Tensor q_fp8, torch::Tensor kv_cache_fp8, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables, int max_model_len,
    int ChunkK, int SplitKV, int num_warps, int TotalCuCount, int version,
    std::optional<torch::Tensor> out_logits_opt
);

PYBIND11_MODULE(moreh_fp8_paged_mqa_logits, m) {
  m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
        "Launch the FP8 Paged MQA Logits HIP kernel with PyTorch tensors.",
        pybind11::arg("q_fp8"),
        pybind11::arg("kv_cache_fp8"),
        pybind11::arg("weights"),
        pybind11::arg("context_lens"),
        pybind11::arg("block_tables"),
        pybind11::arg("max_model_len"),
        pybind11::arg("ChunkK")       = 256,
        pybind11::arg("SplitKV")      = -1,
        pybind11::arg("num_warps")    = 4,
        pybind11::arg("TotalCuCount") = 304,
        pybind11::arg("version")      = 2,   // 2=v2(PAD), 3=v3(PAD+bt-prefetch), 4=v4(swizzle), 5=v5(v3+direct-QW)
        pybind11::arg("out_logits")   = std::optional<torch::Tensor>{}
    );
}