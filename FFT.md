##### L_T with FFT

```
Z_1 = L_T @ A
```

So we just need to compute a product of $L_T$ on any vector of size $T$ using circulant matrices. If we simply pad $L_T$ with $T- 1$ extra zeros on `dim=0` and then also add $T-1$ zeros to `A` we still will have an equivalent product. Let's have a look at a simple example with $T = 3$ and say that `A = [a_1, a_2, a_3]`:

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

##### U_T with FFT

Note that $(U_T)^T = L_T$. Moreover `U_T @ A` can be computed as `A.sum() - L_T @ A + A` with proper reshaping