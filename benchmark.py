import torch
import torch.utils.benchmark as benchmark
import statistics

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

def basic_stats(data: list):
    if data is None:
        return None
    mean_value = round(statistics.mean(data), 6)
    median_value = round(statistics.median(data), 6)
    std_deviation = round(statistics.stdev(data), 6)

    # Other basic statistics
    min_value = round(min(data), 6)
    max_value = round(max(data), 6)

    return f"{mean_value} +/- {std_deviation}, [{min_value}, {median_value}, {max_value}]"


if __name__ == "__main__":
    approach2timing_forward = defaultdict(dict)
    approach2timing_backward = defaultdict(dict)

    Ts = [128, 256, 512, 1024, 2048, 4096]
    N, D = 1, 2048
    for T in tqdm(Ts):
        timing_forward, timing_backward = defaultdict(list), defaultdict(list)
        for _ in range(5):
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
                    timing_forward[approach] += t_forward.timeit(200).times
                except:
                    pass

                t_backward = benchmark.Timer(
                    stmt=stmt_backward,
                    setup=setup,
                    globals=globals_
                )
                try:
                    timing_backward[approach] += t_backward.timeit(200).times
                except:
                    pass
        for approach in approach2setup:
            approach2timing_forward[approach][T] = timing_forward[approach] if timing_forward[approach] else None
            approach2timing_backward[approach][T] = timing_backward[approach] if timing_backward[approach] else None

        print(json.dumps({k: {t: basic_stats(data) for t, data in dict(v).items()} for k, v in approach2timing_forward.items()}, indent=2))
        print(json.dumps({k: {t: basic_stats(data) for t, data in dict(v).items()} for k, v in approach2timing_backward.items()}, indent=2))

    approach2timing_forward = {k: dict(v) for k, v in approach2timing_forward.items()}
    approach2timing_backward = {k: dict(v) for k, v in approach2timing_backward.items()}

    with open("timing_forward.json", "w") as f:
        f.write(json.dumps(approach2timing_forward, indent=2))
    with open("timing_backward.json", "w") as f:
        f.write(json.dumps(approach2timing_backward, indent=2))