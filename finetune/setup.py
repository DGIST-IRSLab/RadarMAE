"""
Build the CUDA ops used by fine-tuning, in place:

    cd finetune && python setup.py build_ext --inplace

Adapted from OpenPCDet's setup.py (https://github.com/open-mmlab/OpenPCDet, Apache-2.0).
Requires a CUDA toolkit whose version matches the one PyTorch was built with.
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# conda-installed CUDA toolkits keep their headers under targets/<arch>/include
CONDA_CUDA_INCLUDE = os.path.join(CUDA_HOME or "", "targets", "x86_64-linux", "include")


def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[f"{module.replace('.', '/')}/src/{s}" for s in sources],
        include_dirs=[CONDA_CUDA_INCLUDE] if os.path.isdir(CONDA_CUDA_INCLUDE) else [],
    )


setup(
    name="radarmae_ops",
    ext_modules=[
        make_cuda_ext("iou3d_nms_cuda", "ops.iou3d_nms",
                      ["iou3d_cpu.cpp", "iou3d_nms_api.cpp", "iou3d_nms.cpp", "iou3d_nms_kernel.cu"]),
        make_cuda_ext("roiaware_pool3d_cuda", "ops.roiaware_pool3d",
                      ["roiaware_pool3d.cpp", "roiaware_pool3d_kernel.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
