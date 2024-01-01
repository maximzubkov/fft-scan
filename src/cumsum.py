import torch


@torch.compile
def pscan_cumsum(A, X):
    N, T, D = X.shape
    device = X.device

    # A_log \in [N x T]
    A_log = torch.log(A.to(dtype=torch.cfloat))

    A_log_flipped = torch.flip(A_log, dims=(1,))
    UA = torch.flip(A_log_flipped.cumsum(dim=-1), dims=(1,))

    W = UA
    W = W.real
    W_max = W.max()
    e_W = torch.exp(W - W_max)
    e_W = e_W.unsqueeze(-1)

    V = -UA + A_log
    V = V.real
    V_max = V.max()
    e_V = torch.exp(V - V_max)
    e_V = e_V.unsqueeze(-1)
    Y_ = e_V * torch.cumsum(e_W * X, dim=1) * (torch.exp(V_max + W_max))

    # After exp we no longer have complex components
    Y_ = Y_.real
    Y_ = torch.cat([torch.zeros(N, 1, D, device=device), Y_[:, :-1, :]], dim=1) 
    Y = Y_ + X
    return Y    