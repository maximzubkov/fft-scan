import torch 

def pscan_naive(A, X):
    N, T, D = X.shape
    Y = torch.zeros(N, T, D)
    Y[:, 0, :] = X[:, 0, :]
    for k in range(1, X.shape[1]):
        Y[:, k, :] = A[:, k - 1].unsqueeze(1) * Y[:, k - 1, :] + X[:, k, :]
    
    return Y
