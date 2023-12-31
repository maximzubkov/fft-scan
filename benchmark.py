import torch
import torch.utils.benchmark as benchmark

from src.naive import pscan_naive
from src.ff import pscan_ff
from src.fft_simple import pscan_fft_simple
from src.fft_efficeint import pscan_fft_efficient

N, T, D = 20, 128, 378

A = torch.randn(N, T).requires_grad_() / 10 + 1
X = torch.randn(N, T, D).requires_grad_() / 1000

t_naive = benchmark.Timer(
    stmt='pscan_naive(A, X)',
    setup='from src.naive import pscan_naive',
    globals={'A': A, 'X': X}
)

t_ff = benchmark.Timer(
    stmt='pscan_ff(A, X)',
    setup='from src.ff import pscan_ff',
    globals={'A': A, 'X': X}
)

t_fft_simple = benchmark.Timer(
    stmt='pscan_fft_simple(A, X)',
    setup='from src.fft_simple import pscan_fft_simple',
    globals={'A': A, 'X': X}
)

t_fft_efficient = benchmark.Timer(
    stmt='pscan_fft_efficient(A, X)',
    setup='from src.fft_efficeint import pscan_fft_efficient',
    globals={'A': A, 'X': X}
)

print(t_naive.timeit(100))
print(t_ff.timeit(100))
print(t_fft_simple.timeit(100))
print(t_fft_efficient.timeit(100))