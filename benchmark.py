import json
import statistics
from collections import defaultdict

import torch
import torch.utils.benchmark as benchmark
from tqdm import tqdm

from src.cumsum import pscan_cumsum
from src.ff import pscan_ff
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
    "ff": "from src.ff import pscan_ff",
    "cumsum": "from src.cumsum import pscan_cumsum",
    "fft_efficient": "from src.fft_efficeint import pscan_fft_efficient",
}

approach2fn = {
    "ff": pscan_ff,
    "cumsum": pscan_cumsum,
    "fft_efficient": pscan_fft_efficient,
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

    return (
        f"{mean_value} +/- {std_deviation}, [{min_value}, {median_value}, {max_value}]"
    )

def init_globals(approach: str, A: torch.Tensor, X: torch.Tensor, Y_init: torch.Tensor):
    globals_ = {
        "A": A.clone().requires_grad_(),
        "X": X.clone().requires_grad_(),
        "pscan_forward": pscan_forward,
        "pscan_backward": pscan_backward,
    }
    if approach == "ff":
        globals_["Y_init"] = Y_init.clone().requires_grad_()
    else:
        globals_["Y_init"] = None
    globals_["fn"] = approach2fn[approach]
    return globals_

if __name__ == "__main__":
    approach2timing_forward = defaultdict(dict)
    approach2timing_backward = defaultdict(dict)

    Ts = [512, 1024, 2048, 3072, 4096, 5120, 6144]
    N, D = 2, 1024
    for T in tqdm(Ts):
        timing_forward, timing_backward = defaultdict(list), defaultdict(list)
        A, X = torch.randn(N, T, device="cuda"), torch.randn(N, T, D, device="cuda")
        Y_init = torch.randn(N, D, device="cuda")
        for approach, approach_setup in tqdm(approach2setup.items()):
            globals_ = init_globals(approach, A, X, Y_init)
            setup = "\n".join([
                approach_setup,
                "A.grad.zero_() if A.grad is not None else None",
                "X.grad.zero_() if X.grad is not None else None",
                "Y_init.grad.zero_() if (Y_init is not None) and (Y_init.grad is not None) else None"
            ]) 

            for measurement in tqdm(range(50)):
                t_forward = benchmark.Timer(
                    stmt=stmt_forward, setup=setup, globals=globals_
                )
                try:
                    timing_forward[approach] += t_forward.timeit(200).times
                except:
                    pass
            
            for measurement in tqdm(range(50)):
                t_backward = benchmark.Timer(
                    stmt=stmt_backward, setup=setup, globals=globals_
                )
                try:
                    timing_backward[approach] += t_backward.timeit(200).times
                except:
                    pass
        for approach in approach2setup:
            approach2timing_forward[approach][T] = (
                timing_forward[approach] if timing_forward[approach] else None
            )
            approach2timing_backward[approach][T] = (
                timing_backward[approach] if timing_backward[approach] else None
            )

        print(
            json.dumps(
                {
                    k: {t: basic_stats(data) for t, data in dict(v).items()}
                    for k, v in approach2timing_forward.items()
                },
                indent=2,
            )
        )
        print(
            json.dumps(
                {
                    k: {t: basic_stats(data) for t, data in dict(v).items()}
                    for k, v in approach2timing_backward.items()
                },
                indent=2,
            )
        )

    approach2timing_forward = {k: dict(v) for k, v in approach2timing_forward.items()}
    approach2timing_backward = {k: dict(v) for k, v in approach2timing_backward.items()}

    with open("timing_forward.json", "w") as f:
        f.write(json.dumps(approach2timing_forward, indent=2))
    with open("timing_backward.json", "w") as f:
        f.write(json.dumps(approach2timing_backward, indent=2))
