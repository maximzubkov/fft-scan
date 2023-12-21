# fft-scan

### 1. Introduction
This repo is inspired by a series of recent posts by François Fleuret on [X](https://twitter.com/francoisfleuret/status/1735907836238954589). The goal is to implement PScan algorithms in a simple yet efficient way.

### 2. Problem
Let's consider tensor $X \in \mathbb{R}^{N \times T \times D}$, and matrix $A \in \mathbb{R}^{N \times T}$. The goal is to compute tensor $Y \in \mathbb{R}^{N \times T \times D}$. Let's denote:
 $$X[:, t, :] \text{ as } X_t$$  $$A[:, t] \text{ as } A_t$$  $$Y[:, t, :] \text{ as } Y_t$$

And let $$Y_0 = X_0$$
And let $Y_t$ can be calculated as follows:

$$Y_t = A_{t - 1} * Y_{t-1} + X_t $$

Where $A_{t - 1} * Y_{t-1}$ satnds for a component-wise product of $A_t$ on the tensor $Y_t$. The goal is to calculate $Y_t$ and ensure that 

### 3. Solution

##### 3.1 Refolmulation
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

##### 3.2 $Z$ matrix

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

So we simplified the task to the calculation of $\overline{Z}$

##### 3.3 FFT

We know that $Z_{i,j}$ = A_i * A_{i-1} * ... A_{j}, let's assume for simplicity that $A_{i, t} > 0 \ \forall i, t$, we will fix this trick later. Then $Z_{i, j} = exp(\sum\limits_{k=j}^i \ln(A_k))$. First lets upper triangular matrix as $U_k$ :

$$U_k = \begin{bmatrix}
1 & 1 & 1 & ...& 1 & 1 \\
0 & 1 & 1 & ... & 1 & 1 \\
\vdots & \vdots &\vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & ... & 1 & 1 \\
0 & 0 & 0 & ... & 0 & 1 \\
\end{bmatrix} \in \mathbb{R}^{k \times k}$$

Now express $\overline{Z}$ as a matrix-vector product, $\forall n$:

$$\overline{Z}[n, t, :t + 1] = U_t \cdot \begin{bmatrix}
\ln(A[n, 0]) \\
\ln(A[n, 1]) \\
\vdots \\
\ln(A[n, t-1]) \\
\end{bmatrix}
$$

$$
\overline{Z}[n, t, t + 1:] = 0
$$

or in PyTorch:
```
D[n, :, :] = torch.cat([U_t, torch.zeros(T - t, T)], dim=0)
Z_[n, t, :] = D[n, :, :] @ torch.log(A)[n, :]
```

Turns out this product can be computed efficiently using FFT. Turns out we can extend matrix $D[n, :, :]$ to make it [Circulant](https://en.wikipedia.org/wiki/Circulant_matrix) and leveraging FFT we can multiply circulant matrix by vector in just $O(T \log (T))$ operartion, there is an [excellent paper](https://arxiv.org/pdf/2103.02605.pdf) about Circulant matices.

##### 3.4 $D$ decomposition

We previously denoted as $U_k$ upper triangular matrix, lets now introduce $L_k$ -- a lower triangular matrix, and $Ld_k = L_k - I_k$ -- a lower triangular matrix with zeros on the diagonal. It's easy to see that:
```
U_k = torch.ones(k) - Ld_k
```

So to compute $Z_[n, t, :]$ we now need to calculate two things:

```
Z_1[n, t, :] = torch.cat([torch.ones(T, t), torch.zeros(T, T-t)]) @ torch.log(A)[n, :]
```

and

```
Z_2[n, t, :] = torch.cat([torch.cat([Ld_t, torch.zeros(T-t, t)], dim=0), torch.zeros(T-t, T)], dim=1) @ torch.log(A)[n, :]
```

so 

```
Z_[n, t, :] = Z_1[n, t, :] - Z_2[n, t, :]
```

The finanl thing left is to represent $Z_1$ and $Z_2$ calculation as a profuct of certain circulant matrix on `torch.log(A)`

##### 3.5 $Z_1$

$Z_1$ is slightly simplier. First let's not that
```
Z_1[n, t, :] = L_T @ torch.log(A[n, :]) # for any t
```

So we just need to compute a product of $L_T$ on any vector of size $T$ using ciruclant matrices. If we simply pad $L_T$ with $T- 1$ extra zeros on `dim=0` and then also add $T-1$ zeros to `torch.log(A[n, :])` we still will have an eqivalent product. Let's have a look at a simple example with $T = 3$ and say that `torch.log(A[n, :]) = [a_1,a_2,a_3]`:

$$
\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & 1 & 1 \\
\end{bmatrix} \cdot \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
\end{bmatrix} = 
$$

For any $x_0, ..., x_5$:

$$
\begin{bmatrix}
1 & 0 & 0 & x_0 & x_1 \\
1 & 1 & 0 & x_2 & x_3 \\
1 & 1 & 1 & x_4 & x_5 \\
\end{bmatrix} \cdot \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
0 \\
0 \\
\end{bmatrix} = 
\left(\begin{bmatrix}
1 & 0 & 0 & 1 & 1 \\
1 & 1 & 0 & 0 & 1 \\
1 & 1 & 1 & 0 & 0 \\
0 & 1 & 1 & 1 & 0 \\
0 & 0 & 1 & 1 & 1 \\
\end{bmatrix} \cdot \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
0 \\
0 \\
\end{bmatrix}\right) [:3, :]
$$

And as we can see on the left side we have a Circulant matrix based on vector `[1, 1, 1, 0,0]`!

##### 3.6 $Z_2$

A very similar thing could be done with each component of $Z_2$, the only difference is that for different $t_1$ and $t_2$, the components $Z_2[n, t_1, :]$ and $Z_2[n, t_2, :]$ will require different ciculant martices, here is an example for $T=4$ and $t = 2$:

$$
\begin{bmatrix}
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{bmatrix} \cdot \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
a_3 \\
\end{bmatrix} = 
$$

For any $x_0, ..., x_5$:

$$
\begin{bmatrix}
0 & 0 & 0 & 0 & x_0 & x_1 & x_2 \\
1 & 0 & 0 & 0 & x_3 & x_4 & x_5 \\
1 & 1 & 0 & 0 & x_6 & x_7 & x_8 \\
0 & 0 & 0 & 0 & x_9 & x_10 & x_11 \\
\end{bmatrix} \cdot \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
a_3 \\
0 \\
0 \\
0 \\
\end{bmatrix}$$

Now let's consider the following product:

$$
\left(
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 1 & 1 \\
1 & 0 & 0 & 0 & 0 & 0 & 1 \\
1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 0 \\
\end{bmatrix}\cdot \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
a_3 \\
0 \\
0 \\
0 \\
\end{bmatrix}\right) [:4, :])
$$

Note that line 3 (counting from 0) of the ciculant matrix is different from line 3 of initial matrix. This can be fixed, it we calculate $Z_2$ using ciculant and after than multiply output by $L_T$ to fix this inconsistency.

##### 3.7 Negative values of matrix $A$

Since FFT can be performed over complex numbers its not a problem, we just need to comput `torch.log` in complex space. Note that $log(-x)$ can be computed using $x \cdot e^{i \cdot \phi} = x \cdot \cos(\phi) + x \cdot i \sin(\phi)= -x$, so $\log(x) = log\(x) + i \pi$,

### 4. Code

To run the code `pytorch` is required.
