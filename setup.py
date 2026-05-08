# setup.py
from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME,
    TORCH_LIB_PATH,
    BuildExtension,
    CUDAExtension,
)


def is_cuda():
    return (
        os.path.exists(CUDA_HOME) and torch.cuda.is_available(
        ) and torch.version.cuda
    )


def is_rocm():
    return os.path.exists(ROCM_HOME) and torch.cuda.is_available() and torch.version.hip


if is_rocm():
    os.environ["CC"] = f"{ROCM_HOME}/bin/hipcc"
    os.environ["CXX"] = f"{ROCM_HOME}/bin/hipcc"
    os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"


if is_rocm():
    setup(
        name='custom_fp8_mqa_logits',
        ext_modules=[
            CUDAExtension('moreh_fp8_paged_mqa_logits',
                          sources=[
                              'csrc/binding.cpp',
                              'csrc/fp8_paged_mqa_logits.cpp',
                          ],
                          library_dirs=[f"{ROCM_HOME}/lib", TORCH_LIB_PATH],
                          runtime_library_dirs=[
                              f"{ROCM_HOME}/lib", TORCH_LIB_PATH],
                          extra_compile_args=[
                              "-O3",
                              "-DNDEBUG",
                              "-std=c++17",
                              f"--offload-arch=gfx942",
                              "-D__HIP_PLATFORM_AMD__=1",
                              "-DUSE_ROCM",
                              "-Rpass-analysis=kernel-resource-usage"
                              # "-D_GLIBCXX_USE_CXX11_ABI=0",
                          ],),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )