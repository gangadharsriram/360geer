"""Numerical-parity check for utils/reloc_utils.py's vectorized compute_relocation_cuda.

Checks the vectorized (P,)-batched implementation against a brute-force,
unvectorized, per-Gaussian Python loop transcribing the exact same formula
(3dgs-mcmc's compute_relocation CUDA kernel, "Equation (9)" in the 3DGS-MCMC
paper) - not the CUDA kernel itself, since the vendored 3dgs-mcmc/ reference
implementation's CUDA extension isn't built in this environment and building
it is a heavyweight, separate step nobody asked for. This still catches any
vectorization bug in reloc_utils.py, since the brute-force version is a
direct, obviously-faithful transcription of the same per-Gaussian math.

Usage: python scripts/test_reloc_pytorch_vs_cuda.py
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.reloc_utils import compute_relocation_cuda, N_MAX, _binoms


def brute_force_relocation(opacity_old, scale_old, N):
    """Unvectorized per-Gaussian transcription of the CUDA kernel, one
    Gaussian at a time, python floats/loops throughout - independent of
    reloc_utils.py's tensor-batched implementation."""
    P = opacity_old.shape[0]
    opacity_new = torch.zeros_like(opacity_old)
    scale_new = torch.zeros_like(scale_old)
    for p in range(P):
        o_old = opacity_old[p].item()
        n = min(max(int(N[p].item()), 1), N_MAX - 1)
        o_new = 1.0 - (1.0 - o_old) ** (1.0 / n)
        denom_sum = 0.0
        for i in range(1, n + 1):
            for k in range(i):
                bin_coeff = _binoms[i - 1, k].item()
                term = ((-1.0) ** k / math.sqrt(k + 1)) * (o_new ** (k + 1))
                denom_sum += bin_coeff * term
        coeff = o_old / denom_sum
        opacity_new[p] = o_new
        scale_new[p] = coeff * scale_old[p]
    return opacity_new, scale_new


def demo():
    torch.manual_seed(0)
    P = 200
    opacity_old = torch.rand(P) * 0.9 + 0.05  # (0.05, 0.95)
    scale_old = torch.rand(P, 3) * 0.5 + 0.01
    N = torch.randint(1, 20, (P,))

    vec_opacity, vec_scale = compute_relocation_cuda(opacity_old, scale_old, N.clone())
    ref_opacity, ref_scale = brute_force_relocation(opacity_old, scale_old, N.clone())

    assert torch.allclose(vec_opacity, ref_opacity, atol=1e-5), \
        f"opacity mismatch: max diff {(vec_opacity - ref_opacity).abs().max().item()}"
    assert torch.allclose(vec_scale, ref_scale, atol=1e-5), \
        f"scale mismatch: max diff {(vec_scale - ref_scale).abs().max().item()}"

    print(f"OK - vectorized compute_relocation_cuda matches brute-force reference "
          f"on {P} random Gaussians (N in [1,19)).")
    print(f"  max opacity diff: {(vec_opacity - ref_opacity).abs().max().item():.2e}")
    print(f"  max scale diff:   {(vec_scale - ref_scale).abs().max().item():.2e}")


if __name__ == "__main__":
    demo()
