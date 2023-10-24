import sys
import warnings
import os
import glob
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    load,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

ext_modules = []
extras = {}

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])


# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ["-DVERSION_GE_1_1"]
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ["-DVERSION_GE_1_3"]
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)

if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")
    raise_if_cuda_home_none("--cuda_ext")
    check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)
    
    ext_modules.append(
        CUDAExtension(
            name="fused_layer_norm_cuda",
            sources=["csrc/layer_norm_cuda.cpp", "csrc/layer_norm_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-maxrregcount=50", "-O3", "--use_fast_math"] + version_dependent_macros,
            },
        )
  )

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]
    
if "--fast_multihead_attn" in sys.argv:
    sys.argv.remove("--fast_multihead_attn")
    raise_if_cuda_home_none("--fast_multihead_attn")

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")

    if bare_metal_version >= Version("11.0"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.1"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_86,code=sm_86")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])
    ext_modules.append(
        CUDAExtension(
            name="fast_multihead_attn",
            sources=[
                "csrc/multihead_attn_cuda.cpp",
                "csrc/self_multihead_attn_bias_additive_mask_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ]
                + version_dependent_macros
                + generator_flag
                + cc_flag,
            },
            include_dirs=[
                os.path.join(this_dir, "csrc/cutlass/include/"),
                os.path.join(this_dir, "csrc/cutlass/tools/util/include")
            ],
        )
    )

setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=("csrc",)
    ),
    install_requires=["packaging>20.6"],
    description="PyTorch Extensions written by NVIDIA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    extras_require=extras,
)