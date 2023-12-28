### 1. Introduction
This repo is inspired by a series of recent posts by François Fleuret on [X](https://twitter.com/francoisfleuret/status/1735907836238954589). The goal is to implement the PScan algorithm in a simple yet efficient way
### 2. Problem
Let's consider tensor $X \in \mathbb{R}^{N \times T \times D}$, and matrix $A \in \mathbb{R}^{N \times T}$. Let's denote:
 $$X[:, t, :] \text{ as } X_t$$  $$A[:, t] \text{ as } A_t$$  $$Y[:, t, :] \text{ as } Y_t$$

And let $$Y_0 = X_0$$
And let $Y_t$ can be calculated as follows:

$$Y_t = A_{t - 1} * Y_{t-1} + X_t $$

Where $A_{t - 1} * Y_{t-1}$ satnds for a component-wise product of $A_t$ on the tensor $Y_t$. The goal is to calculate $Y \in \mathbb{R}^{N \times T \times D}$.

### 3. Solution

#### 3.1 Reformulation
Knowing that $Y_t = A_{t - 1} * Y_{t-1} + X_t$ we can substitute $Y_{t - 1}$ and get 

$$Y_t = X_t + A_{t - 1} * X_{t-1} + A_{t - 1} * A_{t - 2} * Y_{t-1}$$

Following the recursion, for every $t > 0$ and using $\left[ ... \right]$ to group different components of the equation:

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

#### 3.2 $Z$ matrix

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

#### 3.3 FFT

We know that $Z_{i,j} = A_i * A_{i-1} * ... A_{j}$, let's assume for simplicity that $A_{i, t} > 0 \ \forall i, t$, we will fix this trick later. Then $Z_{i, j} = exp(\sum\limits_{k=j}^i \ln(A_k))$. First, let's denote the upper triangular matrix as $U_k$ :

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

Turns out this product can be computed efficiently using FFT. Turns out we can extend matrix $D[n, :, :]$ to make it [Circulant](https://en.wikipedia.org/wiki/Circulant_matrix) and leveraging FFT we can multiply circulant matrix by vector in just $O(T \log (T))$ operation, there is an [excellent paper](https://arxiv.org/pdf/2103.02605.pdf) about Circulant matices.

#### 3.4 Negative values of matrix $A$

Since FFT can be performed over complex numbers it's not a problem, we just need to compute `torch.log` in complex space. Note that $\log(-|x|)$ can be computed using $|x| \cdot e^{i \cdot \phi} = |x| \cdot \cos(\phi) + |x| \cdot i \sin(\phi)= -|x|$, so $\log(-|x|) = \log\(|x|) + i \pi$,

#### 3.5 Naive Implementation

This can be skipped, since later we describe a more efficient way to perform these calculations.

##### 3.5.1 $D$ decomposition

We previously denoted as $U_k$ upper triangular matrix, let's now introduce $L_k$ -- a lower triangular matrix, and $Ld_k = L_k - I_k$ -- a lower triangular matrix with zeros on the diagonal. It's easy to see that:
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

The final thing left is to represent $Z_1$ and $Z_2$ calculation as a product of a certain circulant matrix on `torch.log(A)`

##### 3.5.2 $Z_1$

$Z_1$ is slightly simplier. First, let's note that
```
Z_1[n, t, :] = L_T @ torch.log(A[n, :]) # for any t
```

So we just need to compute a product of $L_T$ on any vector of size $T$ using circulant matrices. If we simply pad $L_T$ with $T- 1$ extra zeros on `dim=0` and then also add $T-1$ zeros to `torch.log(A[n, :])` we still will have an equivalent product. Let's have a look at a simple example with $T = 3$ and say that `torch.log(A[n, :]) = [a_1, a_2, a_3]`:

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

And as we can see on the left side we have a Circulant matrix based on vector `[1, 1, 1, 0, 0]`!

##### 3.5.3 $Z_2$

A very similar thing could be done with each component of $Z_2$, the only difference is that for different $t_1$ and $t_2$, the components $Z_2[n, t_1, :]$ and $Z_2[n, t_2, :]$ will require different circulant matrices, here is an example for $T=4$ and $t = 2$:

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
0 & 0 & 0 & 0 & x_9 & x_{10} & x_{11} \\
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

Note that line 3 (counting from 0) of the circulant matrix is different from line 3 of the initial matrix. This can be fixed if we calculate $Z_2$ using a circulant and then multiply the output by $L_T$ to fix this inconsistency.

#### 3.6 Efficient Implementation

#### 3.6.1 Intuition

Even though $Z$ can be computed efficiently, it still requires $O(N^2)$ to store the tensor in memory and $O(N^2)$ to calculate the product of this tensor with tensor with $X$. So in this section, we describe a more efficient way to perform the calculations. First let's have a look at $Z[n, T - 1, :]$:

$$
Z[n, T - 1, :] = \left(\sum\limits_{k=0}^{T-1} \ln(A[n, k]), \sum\limits_{k=1}^{T-1} \ln(A[n, k]), ..., \sum\limits_{k=T-2}^{T-1} \ln(A[n, k])\right)^T
$$

or, in other words:

$$
Z[n, T - 1, :] =  U_{T} \cdot \begin{bmatrix}
\ln(A[n, 0]) \\
\ln(A[n, 1]) \\
\vdots \\
\ln(A[n, T-1]) \\
\end{bmatrix}
$$

As we showed earlier, the product of any vector on the matrix $U_T$ can be computed efficiently using FFT using matrices $Z_1$ and $Z_2$. Now let's note the following fact, for any $t$:

```
Z[n, t - 2, :] = (Z[n, t - 1, :] - ln(A[n, t - 1])) * torch.cat([torch.ones(t), torch.zeros(T-t)])
```

So this leads to the conclusion that knowing $Z[n, T - 1, :]$ we can easily calculate $Z[n, t, :]$ for any $t$ by gradually subtracting $\ln(A[n, k])$ and using a proper mask `torch.cat([torch.ones(t), torch.zeros(T-t)])`. Let's now rewrite this in a tensor form. Lets denote $Z[n, T - 1, :]$ as $V$

$$
Y = X + \text{exp}(Z) \cdot X = X + (L_T * \text{exp}(\overline{Z})) \cdot X
$$

where 

$$
\overline{Z} = \begin{bmatrix}
V_0, V_1, ..., V_T \\
V_0, V_1, ..., V_T \\
\vdots \\
V_0, V_1, ..., V_T \\
\end{bmatrix} - \begin{bmatrix}
0 & 1 & 1 & ...& 1 & 1 \\
0 & 0 & 1 & ... & 1 & 1 \\
\vdots & \vdots &\vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & ... & 1 & 1 \\
0 & 0 & 0 & ... & 0 & 1 \\
0 & 0 & 0 & ... & 0 & 0 \\
\end{bmatrix} \cdot \begin{bmatrix}
\ln(A[n, 0]) & \ln(A[n, 0]) & ... & \ln(A[n, 0])\\
\ln(A[n, 1]) & \ln(A[n, 1]) & ... & \ln(A[n, 1]) \\
\vdots \\
\ln(A[n, T-1]) & \ln(A[n, T-1]) & ... & \ln(A[n, T-1])\\
\end{bmatrix} =
$$

or 

$$
\begin{bmatrix}
V_0, V_1, ..., V_T \\
V_0, V_1, ..., V_T \\
\vdots \\
V_0, V_1, ..., V_T \\
\end{bmatrix} - (U_T - I) \cdot \begin{bmatrix}
\ln(A[n, 0]) & \ln(A[n, 0]) & ... & \ln(A[n, 0])\\
\ln(A[n, 1]) & \ln(A[n, 1]) & ... & \ln(A[n, 1]) \\
\vdots \\
\ln(A[n, T-1]) & \ln(A[n, T-1]) & ... & \ln(A[n, T-1])\\
\end{bmatrix} = \begin{bmatrix}
V^T \\
V^T \\
\vdots \\
V^T \\
\end{bmatrix} - \begin{bmatrix}
W &
W &
... &
W &
\end{bmatrix}
$$

Moreover, we know that we can compute the product of $U_T$ with any vector efficiently, so both $V$ and $W$ can be computed using FFT.

#### 3.6.2 $V$ and $W$

We know that both $V$ and $W$ can be calculated efficiently, the only thing remaining is to compute $Y$ efficiently. Knowing that both $V$ and $W$ have shape $N \times T$ we can rewrite the equating the following way:

```
Y = X + (L_T * torch.exp(V.unsqueeze(1) - W.unsqueeze(2))) @ X
```

Let's have a closer look for any fixed $n$:

$$
L_T * \begin{bmatrix}
e^{V_0} \cdot e^{-W_0} & e^{V_1} \cdot e^{-W_0} & ... & e^{V_{T-1}} \cdot e^{-W_0} \\
e^{V_0} \cdot e^{-W_1} & e^{V_1} \cdot e^{-W_1} & ... & e^{V_{T-1}} \cdot e^{-W_1} \\
\vdots \\
e^{V_0} \cdot e^{-W_{T-1}} & e^{V_1} \cdot e^{-W_{T-1}} & ... & e^{V_{T-1}} \cdot e^{-W_{T-1}} \\
\end{bmatrix} = \begin{bmatrix}
e^{V_0} \cdot e^{-W_0} & 0 & ... & 0 \\
e^{V_0} \cdot e^{-W_1} & e^{V_1} \cdot e^{-W_1} & ... & 0 \\
\vdots \\
e^{V_0} \cdot e^{-W_{T-1}} & e^{V_1} \cdot e^{-W_{T-1}} & ... & e^{V_{T-1}} \cdot e^{-W_{T-1}} \\
\end{bmatrix}
$$

or 

$$
\exp(V)^T * L_T * \exp(-W)
$$

The final step is to use the following trick. For any vector $a$ and matrix $B$: $a^T * B = \text{diag}(a) \cdot B$, and $B * a = B \cdot \text{diag}(a)$, so we get a very elegant formula for $Y$:

$$
Y = X + \text{diag}(e^{V}) \cdot L_T \cdot \text{diag}(e^{W}) \cdot X = X + (e^{V})^T * (L_T \cdot (e^{W} * X))
$$

$L_T$, as well as $U_T$, can be multiplied by vector efficiently, requiring only $O(T \log(T))$ operations to perform this opperation. So we need $O(T \log(T))$ to calcualte $V$ and $W$, then $e^{W} * X$ requires $O(T)$ operations; $L_T \cdot (...)$ brings another $O(T \log(T))$ and finally, $X + (e^{V})^T * (...)$ requires another $O(T)$, so the total compelxity of this approach is $O(T \log(T))$

### 4. Code

To run the code `pytorch` is required.
