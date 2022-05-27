import glob
import os.path as osp

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

_ext_src_root = osp.join(osp.dirname(__file__), "ext")
_ext = load(
    "_ext",
    sources=glob.glob(osp.join(_ext_src_root, "src", "*.cpp"))
    + glob.glob(osp.join(_ext_src_root, "src", "*.cu")),
    extra_include_paths=[osp.join(_ext_src_root, "include")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
    with_cuda=True,
    verbose=True,
)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        out = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(out)
        return out


furthest_point_sample = FurthestPointSampling.apply
