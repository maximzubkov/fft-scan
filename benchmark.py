import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity

from collections import defaultdict
import json
from tqdm import tqdm

from src.naive import pscan_naive
from src.ff import pscan_ff
from src.cumsum import pscan_cumsum
from src.fft_simple import pscan_fft_simple
from src.fft_efficeint import pscan_fft_efficient

def pscan_forward(fn, A, X, Y_init=None):
    Y = fn(A, X) if Y_init is None else fn(A, X, Y_init)

stmt_forward = "pscan_forward(fn, A, X, Y_init)"

def pscan_backward(fn, A, X, Y_init=None):
    Y = fn(A, X) if Y_init is None else fn(A, X, Y_init)
    loss = Y.sum()
    loss.backward(retain_graph=True)

stmt_backward = "pscan_backward(fn, A, X, Y_init)"

approach2setup = {
    'naive': 'from src.naive import pscan_naive',
    'ff': 'from src.ff import pscan_ff',
    'cumsum': 'from src.cumsum import pscan_cumsum',
    'fft_simple': 'from src.fft_simple import pscan_fft_simple',
    'fft_efficient': 'from src.fft_efficeint import pscan_fft_efficient',
}

approach2fn = {
    'naive': pscan_naive,
    'ff': pscan_ff,
    'cumsum': pscan_cumsum,
    'fft_simple': pscan_fft_simple,
    'fft_efficient': pscan_fft_efficient,
}


if __name__ == "__main__":
    approach2timing_forward = defaultdict(dict)
    approach2timing_backward = defaultdict(dict)

    Ts = [128, 256, 512, 1024, 2048, 4096, 8192]
    N, D = 2, 2048
    for T in tqdm(Ts):
        timing_forward, timing_backward = [], []
        for _ in range(10):
            globals_ = {
                'A': torch.randn(N, T, device='cuda').requires_grad_() / 10 + 1,
                'X': torch.randn(N, T, D, device='cuda').requires_grad_() / 10,
                'pscan_forward': pscan_forward,
                'pscan_backward': pscan_backward
            }
            for approach, setup in tqdm(approach2setup.items()):
                globals_["fn"] = approach2fn[approach]
                if approach == "ff":
                    globals_["Y_init"] = torch.randn(N, D, device='cuda').requires_grad_() / 10
                else:
                    globals_["Y_init"] = None

                t_forward = benchmark.Timer(
                    stmt=stmt_forward,
                    setup=setup,
                    globals=globals_
                )
                try:
                    timing_forward += t_forward.timeit(50).times
                except:
                    pass

                t_backward = benchmark.Timer(
                    stmt=stmt_backward,
                    setup=setup,
                    globals=globals_
                )
                try:
                    timing_backward += t_backward.timeit(50).times
                except:
                    pass

        approach2timing_forward[approach][T] = timing_forward if timing_forward else None
        approach2timing_backward[approach][T] = timing_backward if timing_backward else None

        print(json.dumps({k: dict(v) for k, v in approach2timing_forward.items()}, indent=2))
        print(json.dumps({k: dict(v) for k, v in approach2timing_backward.items()}, indent=2))

    approach2timing_forward = {k: dict(v) for k, v in approach2timing_forward.items()}
    approach2timing_backward = {k: dict(v) for k, v in approach2timing_backward.items()}

    with open("timing_forward.json", "w") as f:
        f.write(json.dumps(approach2timing_forward, indent=2))
    with open("timing_backward.json", "w") as f:
        f.write(json.dumps(approach2timing_backward, indent=2))