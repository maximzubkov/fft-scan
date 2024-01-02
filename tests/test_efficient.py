import torch

from src.fft_efficeint import pscan_fft_efficient

N, T, D = 20, 128, 378


def test_pscan_fft_efficient():
    A = torch.randn(N, T).requires_grad_() / 10 + 1
    X = torch.randn(N, T, D).requires_grad_() / 1000

    Y_fft = pscan_fft_efficient(A, X)

    Y_expected = torch.zeros(N, T, D)
    Y_expected[:, 0, :] = X[:, 0, :]
    for k in range(1, X.shape[1]):
        Y_expected[:, k, :] = (
            A[:, k - 1].unsqueeze(1) * Y_expected[:, k - 1, :] + X[:, k, :]
        )

    assert torch.norm(Y_fft - Y_expected).item() < 1e-4
