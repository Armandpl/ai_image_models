import time

import timm
import torch
from tqdm import trange

torch.set_grad_enabled(False)

T4_FLOPS_ADVERTISED = 65e12 # fp16
model = timm.create_model('caformer_m36', pretrained=False)
img = torch.randn(1, 3, 224, 224)

with torch.profiler.profile(with_flops=True) as prof:
   model(img)

n1_flops = sum(avg.flops for avg in prof.key_averages())


BS = 8 # increase to amortize cpu overhead
STEPS = 10
device = 'cuda'
batch = torch.randn(BS, 3, 224, 224).to(device)
model = model.to(device)

st = time.perf_counter()
with torch.autocast(device_type=device, dtype=torch.float16):
    for _ in trange(STEPS):
        model(batch)
torch.cuda.synchronize()
elapsed = time.perf_counter() - st

total_flops = n1_flops * STEPS * BS # number of floating point operation
total_flops /= elapsed # per second

MFU = total_flops/T4_FLOPS_ADVERTISED*100
print(f'\nMFU {MFU:.2f}%')
