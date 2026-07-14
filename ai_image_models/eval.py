import torch


def fid(f1, f2):
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = torch.cov(f1.T), torch.cov(f2.T)
    eig = torch.linalg.eigvals(c1 @ c2).real.clamp(min=0)
    return (((m1 - m2) ** 2).sum() + torch.trace(c1 + c2) - 2 * eig.sqrt().sum()).item()
