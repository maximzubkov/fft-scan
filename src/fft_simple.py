import torch


def pscan_fft_simple(A, X):
    N, T, D = X.shape
    device = X.device

    # A_log \in [N x T]
    A_log = torch.log(A.to(dtype=torch.cfloat))
    # A_log_T \in [T x N]
    A_log_T = A_log.T
    # A_log_T \in [(2T - 1) x N]
    A_log_T = torch.cat([A_log_T, torch.zeros(T - 1, N, device=device)], dim=0)

    # For T = 3
    # mask1 = [1, 1, 1, 0, 0]
    # circulant_matrix = [
    #    [1, 0, 0, 1, 1],
    #    [1, 1, 0, 0, 1],
    #    [1, 1, 1, 0, 0],
    #    [0, 1, 1, 1, 0],
    #    [0, 0, 1, 1, 1],
    # ]
    mask1 = torch.where(
        (torch.arange(2 * T - 1, device=device) <= T - 1),
        1,
        0,
    )
    mask1 = mask1.unsqueeze(1)
    Z1_log_rev = torch.fft.ifft(
        torch.fft.fft(mask1, dim=0) * torch.fft.fft(A_log_T, dim=0), n=2 * T - 1, dim=0
    )
    # Since we add T - 1 of padding zeros to A_log_T
    Z1_log_rev = Z1_log_rev[:T, :].T.unsqueeze(-1)

    # For T = 4 and t = 2
    # mask2[0] = [0, 0, 0, 0, 0, 0, 0]
    # mask2[1] = [0, 1, 0, 0, 0, 0, 0]
    # mask2[2] = [0, 1, 1, 0, 0, 0, 0]
    # mask2[3] = [0, 1, 1, 1, 0, 0, 0]
    #
    # for t = 2
    # circulant_matrix = [
    #    [0, 0, 0, 0, 0, 1, 1],
    #    [1, 0, 0, 0, 0, 0, 1],
    #    [1, 1, 0, 0, 0, 0, 0],
    #    [0, 1, 1, 0, 0, 0, 0],
    #    [0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 1, 0, 0, 0],
    #    [0, 0, 0, 1, 1, 0, 0],
    # ]
    mask2 = torch.where(
        torch.cat(
            [
                (
                    (torch.arange(2 * T - 1, device=device) >= 1)
                    & (torch.arange(2 * T - 1, device=device) <= t)
                ).unsqueeze(0)
                for t in range(T)
            ],
            dim=0,
        ),
        1,
        0,
    )
    mask2 = mask2.unsqueeze(-1)
    Z2_log_rev = torch.fft.ifft(
        torch.fft.fft(mask2, dim=1) * torch.fft.fft(A_log_T.unsqueeze(0), dim=1),
        n=2 * T - 1,
        dim=1,
    )
    # Since we add T - 1 of padding zeros to A_log_T
    Z2_log_rev = Z2_log_rev[:, :T, :]
    Z2_log_rev = Z2_log_rev.permute(2, 0, 1)
    # Fixing the problem casued by line 3 in the example
    Z2_log_rev = torch.tril(Z2_log_rev, diagonal=0)

    Z_log = Z1_log_rev - Z2_log_rev
    # Z \in [N x T x T]
    Z = torch.tril(torch.exp(Z_log), diagonal=0)
    # After exp we no longer have complex components
    Z = Z.real
    # Y \in [N x T x D] = bmm([N x T x T], [N x T x D])
    Y_ = torch.bmm(Z, X)
    Y_ = torch.cat([torch.zeros(N, 1, D, device=device), Y_[:, :-1, :]], dim=1)
    Y = Y_ + X
    return Y
