import math
import torch

# Pure-PyTorch port of 3DGS-MCMC's compute_relocation CUDA kernel
# (3dgs-mcmc/submodules/diff-gaussian-rasterization/cuda_rasterizer/utils.cu,
# "Equation (9) in 3D Gaussian Splatting as Markov Chain Monte Carlo").
# No gradient needed (relocation runs under torch.no_grad() in train.py) and
# N is bounded (<=50), so a vectorized double loop over (i, k) - same shape
# as the CUDA kernel's per-thread loop, batched over the P-Gaussian dimension
# instead of a CUDA grid - is simple and fast enough at this call frequency
# (once per densification_interval, on a few thousand Gaussians at a time).
# See scripts/test_reloc_pytorch_vs_cuda.py for a numerical-parity check
# against the vendored CUDA kernel in 3dgs-mcmc/.

N_MAX = 51
# Pascal's triangle, not math.comb (needs Python 3.8+; this repo's conda env is 3.7).
_binoms = torch.zeros((N_MAX, N_MAX))
_binoms[0, 0] = 1.0
for _n in range(1, N_MAX):
    _binoms[_n, 0] = 1.0
    for _k in range(1, _n + 1):
        _binoms[_n, _k] = _binoms[_n - 1, _k - 1] + (_binoms[_n - 1, _k] if _k <= _n - 1 else 0.0)


def compute_relocation_cuda(opacity_old, scale_old, N):
    """opacity_old, scale_old: rendered-opacity-space / activated-scale-space
    tensors for the Gaussians being relocated (P,) and (P,3). N: (P,) int
    tensor, total copy count per Gaussian (>=1). Returns (opacity_new,
    scale_new) in the same (rendered opacity / activated scale) space -
    callers are responsible for converting back to raw parameter space."""
    N = N.clamp(min=1, max=N_MAX - 1)
    binoms = _binoms.to(opacity_old.device)

    opacity_new = 1.0 - (1.0 - opacity_old).pow(1.0 / N.float())

    denom_sum = torch.zeros_like(opacity_old)
    for i in range(1, N_MAX):
        mask = (N >= i).float()
        if mask.sum() == 0:
            continue
        for k in range(i):
            bin_coeff = binoms[i - 1, k]
            term = ((-1.0) ** k / math.sqrt(k + 1)) * opacity_new.pow(k + 1)
            denom_sum = denom_sum + bin_coeff * term * mask

    coeff = opacity_old / denom_sum
    scale_new = coeff.unsqueeze(-1) * scale_old
    return opacity_new, scale_new
