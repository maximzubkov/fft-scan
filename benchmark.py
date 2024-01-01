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
    approach2memory_forward = defaultdict(dict)
    approach2memory_backward = defaultdict(dict)

    Ts = [128, 256, 512, 1024, 2048, 4096, 8192]
    N, D = 4, 1024
    for T in tqdm(Ts):
        globals_ = {
            'A': torch.randn(N, T, device='cuda').requires_grad_() / 10 + 1,
            'X': torch.randn(N, T, D, device='cuda').requires_grad_() / 10,
            'pscan_forward': pscan_forward,
            'pscan_backward': pscan_backward
        }
        for approach, setup in tqdm(approach2setup.items()):
            globals_["fn"] = approach2fn[approach]
            if approach == "ff":
                globals_["Y_init"] = torch.randn(N, D, device='cuda').requires_grad_() / 1000
            else:
                globals_["Y_init"] = None

            t_forward = benchmark.Timer(
                stmt=stmt_forward,
                setup=setup,
                globals=globals_
            )
            try:
                timing_forward = t_forward.timeit(100).median
            except:
                timing_forward = None

            t_backward = benchmark.Timer(
                stmt=stmt_backward,
                setup=setup,
                globals=globals_
            )
            try:
                timing_backward = t_backward.timeit(100).median
            except:
                timing_backward = None

            approach2timing_forward[approach][T] = timing_forward
            approach2timing_backward[approach][T] = timing_backward

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                pscan_forward(**globals)

            stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=1)
            parts = stats.split("\n")[-5].split()
            memory_forward = -float(parts[-3])
            if parts[-2] == "Mb":
                memory_forward = memory_forward / 1000

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                pscan_backward(**globals)

            stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=1)
            parts = stats.split("\n")[-5].split()
            memory_backward = -float(parts[-3])
            if parts[-2] == "Mb":
                memory_backward = memory_backward / 1000

            approach2memory_forward[approach][T] = memory_forward
            approach2memory_backward[approach][T] = memory_backward

        print(json.dumps({k: dict(v) for k, v in approach2timing_forward.items()}, indent=2))
        print(json.dumps({k: dict(v) for k, v in approach2timing_backward.items()}, indent=2))
        print(json.dumps({k: dict(v) for k, v in approach2memory_forward.items()}, indent=2))
        print(json.dumps({k: dict(v) for k, v in approach2memory_backward.items()}, indent=2))
    approach2timing_forward = {k: dict(v) for k, v in approach2timing_forward.items()}
    approach2timing_backward = {k: dict(v) for k, v in approach2timing_backward.items()}
    approach2memory_forward = {k: dict(v) for k, v in approach2memory_forward.items()}
    approach2memory_backward = {k: dict(v) for k, v in approach2memory_backward.items()}

    with open("timing_forward.json", "w") as f:
        f.write(json.dumps(approach2timing_forward, indent=2))
    with open("timing_backward.json", "w") as f:
        f.write(json.dumps(approach2timing_backward, indent=2))
    with open("memory_forward.json", "w") as f:
        f.write(json.dumps(approach2memory_forward, indent=2))
    with open("memory_backward.json", "w") as f:
        f.write(json.dumps(approach2memory_backward, indent=2))