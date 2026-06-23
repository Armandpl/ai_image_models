import math

import torch
import torch.nn as nn


def time_embed(t, dim=128):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / half))
    args = t[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class FlowMLP(nn.Module):
    def __init__(self, img_shape=(16, 16, 3), t_dim=128, h=1024):
        super().__init__()
        self.img_shape = img_shape
        self.dim = math.prod(img_shape)
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(self.dim + t_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, self.dim),
        )

    def forward(self, z, t):
        return self.net(torch.cat([z, time_embed(t, self.t_dim)], dim=-1))


class SpriteClassifier(nn.Module):
    def __init__(self, img_shape=(16, 16, 3), n_classes=5, h=256):
        super().__init__()
        self.dim = math.prod(img_shape)
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
        )
        self.head = nn.Linear(h, n_classes)

    def forward(self, x):
        return self.head(self.body(x))

    @torch.no_grad()
    def features(self, x):
        device = next(self.parameters()).device
        return self.body(x.to(device)).cpu()
