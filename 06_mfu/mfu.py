import time

import torch

from tqdm import trange

N = 128 # try increasing to ammortize python overhead
A, B = torch.randn(N,N, dtype=torch.float16), torch.randn(N,N, dtype=torch.float16)
MAC_M4_ADVERTISED_FLOPS_FP16 = 8.52

st = time.perf_counter()
with torch.profiler.profile(with_flops=True) as prof:
  for _ in trange(100):
    torch.mm(A, B)
et = time.perf_counter()
n_flops = sum(avg.flops for avg in prof.key_averages())

tflops_per_seconds = n_flops / (et-st) / 1e12

MFU = tflops_per_seconds / MAC_M4_ADVERTISED_FLOPS_FP16
print(f"MFU: {MFU*100:.2f}%")
