# fft-scan

### Introduction
This repo is inspired by a series of recent posts by François Fleuret on [X](https://twitter.com/francoisfleuret/status/1735907836238954589). The goal is to implement PScan algorithms in a simple yet efficient way.

### Problem
Let's consider tensor $X \in \mathbb{R}^{N \times T \times D}$, and matrix $A \in \mathbb{R}^{N \times T}$. The goal is to compute tensor $Y \in \mathbb{R}^{N \times T \times D}$. Let's denote:
 $$X[:, t, :] \text{ as } X_t$$  $$A[:, t] \text{ as } A_t$$  $$Y[:, t, :] \text{ as } Y_t$$

And let $$Y_0 = X_0$$
And let $Y_t$ can be calculated as follows:

$$Y_t = A_{t - 1} * Y_{t-1} + X_t $$

Where $A_{t - 1} * Y_{t-1}$ satnds for a component-wise product of $A_t$ on the tensor $Y_t$. The goal is to calculate $Y_t$ and ensure that 

### Solution

##### Refolmulation
Knowing that $Y_t = A_{t - 1} * Y_{t-1} + X_t$ we can substitute $Y_{t - 1}$ and get 

$$Y_t = X_t + A_{t - 1} * X_{t-1} + A_{t - 1} * A_{t - 2} * Y_{t-1}$$

Following the recustion, for every $t > 0$ and using $\left[ ... \right]$ to group different components of the equation:

$$Y_t = \left[ X_t \right] + \left[ A_{t - 1} * X_{t-1} \right] + \left[ A_{t - 1} * A_{t - 2} * X_{t-1} \right] + ... +  \left[ A_{t - 1} * A_{t - 2} * ... * A_0 * X_0 \right] $$

Now lets denote $A_i * A_{i-1} * ... * A_{j}$ as $Z_{i, j}$, then we can rewrite the above equation $\forall n \in \left[ 0, 1, ..., N - 1\right]$:

$$ \ Y[n, :, :] = X[n, :, :] + \begin{bmatrix}
0 & 0 & ... & 0 & 0 & 0 \\
Z_{0,0} & 0 & ... & 0 & 0 & 0\\
Z_{1,0} & Z_{1, 1} & ... & 0 & 0 & 0 \\
\vdots & \vdots & ... & \vdots  & \vdots  & \vdots \\
Z_{T-3,0} & Z_{T-3, 1} & ... & Z_{T-3, T-3} &  0 & 0\\
Z_{T-2,0} & Z_{T-2, 1} & ... & Z_{T-2, T-3} &  Z_{T-2, T-2} & 0\\
\end{bmatrix} \cdot X[n, :, :]$$

Or using `torch` notation:

```
Y = X + torch.bmm(M, X)
```

So the only thing needed is to compute matrix $M$

##### $Z$ matrix

First, note the first row of $M$ is composed of only $0$ so lets consider matrix $Z$ such that:

```
M = torch.cat([torch.zeros(N, 1, D), Z], dim=1)
```
Now, let's have a closer look at $Z$:

$$ Z = \begin{bmatrix}
Z_{0,0} & 0 & ... & 0 & 0 & 0\\
Z_{1,0} & Z_{1, 1} & ... & 0 & 0 & 0 \\
\vdots & \vdots & ... & \vdots  & \vdots  & \vdots \\
Z_{T-3,0} & Z_{T-3, 1} & ... & Z_{T-3, T-3} &  0 & 0\\
Z_{T-2,0} & Z_{T-2, 1} & ... & Z_{T-2, T-3} &  Z_{T-2, T-2} & 0\\
\end{bmatrix} $$

Then $Z$ can be computed using a matrix $\overline{Z}$:

$$ \overline{Z} = \begin{bmatrix}
Z_{0,0} & Z_{0, 1} & ...& Z_{0, T-1}\\
Z_{1,0} & Z_{1, 1} & ... & Z_{1, T-1} \\
\vdots & \vdots & ... & \vdots \\
Z_{T-1,0} & Z_{1, 1} & ... & Z_{T-1, T-1} \\
\end{bmatrix} $$

Then
```
Z = torch.tril(Z_, diagonal=0)[:, :-1, :]
```

So we simplified the task to calculation of $\overline{Z}$

##### FFT

We know that $Z_{i,j}$ = A_i * A_{i-1} * ... A_{j}, let's assume for simplicity that $A_{i, t} > 0 \ \forall i, t$, we will fix this trick later. Then $Z_{i, j} = exp(\sum\limits_{k=j}^i \ln(A_k))$. Or as a matrix-vector product, $\forall n$:

$$\overline{Z}[n, t, :] = \begin{bmatrix}
1 & 0 & 0& ...& 0 & 0 \\
1 & 1 & 0 & ... & 0 & 0 \\
\vdots & \vdots &\vdots & \vdots & \vdots \\
1 & 1 & 1 & ... & 1 & 0 \\
1 & 1 & 1 & ... & 1 & 1 \\
\end{bmatrix}\ \cdot \begin{bmatrix}
\ln(A[n, 0]) \\
\ln(A[n, 1]) \\
\vdots \\
\ln(A[n, t-1]) \\
\end{bmatrix} = 
$$

$$
\begin{bmatrix}
1 & 0 & 0 & ... & 0 & 0 & 0 & ... & 0 \\
1 & 1 & 0 & ... & 0 & 0 & 0 & ... & 0 \\
\vdots & \vdots &\vdots & \vdots  & \vdots  & \vdots & \vdots & \vdots \\
1 & 1 & 1 & ... & 1 & 0 & 0 & ... & 0 \\
1 & 1 & 1 & ... & 1 & 1 & 0 & ... & 0 \\
\end{bmatrix} \cdot \begin{bmatrix}
\ln(A[n, 0]) \\
\ln(A[n, 1]) \\
\vdots \\
\ln(A[n, t-1])\\
\vdots \\
\ln(A[n, T-1])\\
\end{bmatrix} D[n, :, :] \cdot \ln{A[n, :]}$$

Turns out this porduct can be computed efficiently using FFT. We can pad 



