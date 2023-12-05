import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler

import cutlass_quant


# This is wrong for now
def pack(weight):
    """
    Arguments:
        weight: (in_features, out_features), in int8
    Returns:
        packed_weight: (in_features, out_features / 8)
    """
    din, dout = weight.shape
    assert dout % 8 == 0
    assert weight.dtype == torch.int8
    assert torch.logical_and(weight <= 7, weight >= -8).all()
    # [e0, e1, e2, e3, e4, e5, e6, e7] -> [e0, e2, e4, e6, e1, e3, e5, e7]
    weight = weight.reshape(din, dout // 8, 8).to(torch.int32)
    packed_weight = torch.zeros(din, dout // 8, dtype=torch.int32, device=weight.device)
    packed_weight = (
        weight[:, :, 0]
        | (weight[:, :, 2] << 4)
        | (weight[:, :, 4] << 8)
        | (weight[:, :, 6] << 12)
        | (weight[:, :, 1] << 16)
        | (weight[:, :, 3] << 20)
        | (weight[:, :, 5] << 24)
        | (weight[:, :, 7] << 28)
    )
    return packed_weight


torch.manual_seed(0)
repeats = 30
batch_size = 1
seqlen = 1
# nheads = 32
nheads = 40
# nheads = 64
headdim = 128
n = nheads * headdim
dtype = torch.float16
device = 'cuda'

din = 128
dout = 128
x = torch.eye(din, dtype=dtype, device=device)
w = torch.zeros(din, dout, dtype=torch.int8, device=device)
wpacked = pack(w)
wscale = torch.ones(dout, dtype=dtype, device=device)
y = cutlass_quant.fwd(x, wpacked, wscale, 4)
# breakpoint()
