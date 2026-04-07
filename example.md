## Manual Computation
Let's walk through this example to verify that the algorithm is correct. The input, weight, and target are purposefully constructed such that the first and last output contribute mainly to the result. 

```python
x = Tensor(np.array([[1.0, 0.0, 1.0]]), _name="x") # (1,3)
W = Tensor(np.array([
    [0.6, 0.2],
    [0.3, 0.4],
    [0.2, 0.7]
]), _name="W") # (3,2)
b = Tensor(np.array([[0.1, 0.2]]), _name="b") # (1,2)
y = Tensor(np.array([[1.2, 1.3]]), _name="y")

y_hat = x @ W + b # [0.8+0.1,0.9+0.2] = [0.9,1.1]
print("y hat", y_hat)

loss = (y - y_hat).sum() # (1.2-0.9) + (1.3-1.1) = 0.5
loss.backward()
print("loss", loss)
print("x.grad", x.grad)
print("W.grad", W.grad)
print("b.grad", b.grad)
```

First, recall the formulas for the gradients for a linear layer $Y=XW+B$ where $X \in \mathbb{R}^{b \times n}$, $W \in \mathbb{R}^{n \times m}$, $B \in \mathbb{R}^{b \times m}$, $Y \in \mathbb{R}^{b \times m}$:
1. $\frac{\delta L}{\delta X}=\frac{\delta L}{\delta Y} W^T$
2. $\frac{\delta L}{\delta W}=X^T \frac{\delta L}{\delta Y}$
3. $\frac{\delta L}{\delta b}=\frac{\delta L}{\delta Y}$

Note that B can also be stored as $\mathbb{R}^{1 \times m}$ and then broadcast (copied $b$ times) across the batch dimension but that's not done here. 

$$\frac{\delta L}{\delta X}=\begin{bmatrix}
-1.0 & -1.0
\end{bmatrix} \begin{bmatrix}
0.6 & 0.3 & 0.2 \\
0.2 & 0.4 & 0.7
\end{bmatrix}=\begin{bmatrix}
-0.8 & -0.7 & -0.9
\end{bmatrix}$$

$$\frac{\delta L}{\delta W}=\begin{bmatrix}
1.0 \\
0.0 \\
1.0
\end{bmatrix}\begin{bmatrix}
-1.0 & -1.0
\end{bmatrix}=\begin{bmatrix}
-1.0 & -1.0 \\
0.0 & 0.0 \\
-1.0 & -1.0
\end{bmatrix}$$

$$\frac{\delta L}{\delta b}=\begin{bmatrix}
-1.0 & -1.0
\end{bmatrix}$$

Interpreting matrix derivatives, such as $\frac{\delta L}{\delta X}$: nudging the $i,j$th component of $X$ by $\epsilon$ results in an *additive* contribution of $\epsilon \cdot \left( \frac{\delta L}{\delta X} \right)_{i,j}$ to the loss. Derivatives of scalars (e.g. $L$) w.r.t matrices are always the same shape as the matrices. 

Here, nonzero entries for all derivatives are **all negative** because increasing any of the values (either weights or data entries) would push down the loss since we're undershooting -- $\hat{y} < y$. The second row of the weight matrix is all zero because unilaterally nudging the weight doesn't influence the loss b/c $x_{1}=0$. 

## Algorithm Computation
In our algorithm, we follow the general structure:
- In the forwards pass, compute a DAG with all the operations + values
- In the backwards pass, compute a topological ordering of said graph and traverse, computing the partial derivatives w.r.t loss along the way

Going line, by line:
```python
x = Tensor(np.array([[1.0, 0.0, 1.0]]), _name="x") # (1,3)
W = Tensor(np.array([
    [0.6, 0.2],
    [0.3, 0.4],
    [0.2, 0.7]
]), _name="W") # (3,2)
b = Tensor(np.array([[0.1, 0.2]]), _name="b") # (1,2)
y = Tensor(np.array([[1.2, 1.3]]), _name="y")
``` 

instantiates all our tensors to be used for calculation. Next, the line `y_hat = x @ W + b` works recursively. No gradients are updated yet. Notation a bit imprecise for simplicity. 
1. `x @ W` -> calls __matmul__ method. Produces `out` with children `x, W` and operation `@`
2. `out + b` -> calls __add__ method. Produces `y_hat` with children `out, b` and operation `+`

For the line `loss = (y - y_hat).sum()` (leaving out children, operation for simplicity):
1. `y-y_hat` -> calls __sub__ method, which then calls __neg__ on y_hat (producing `y_hat'`), then __add__ producing `out`
2. `out.sum()` -> calls `sum()` producing `loss`

When we call `loss.backward()`, it uses a topological sort which guarantees for every edge `u,v` that `u` comes before `v` in the graph. This works by doing a dfs and only appending the current node once its neighbors have all been visited. We reverse this order for backprop. The gradients are therefore updated from back to front. 

Let's look at the output:
```
visited Tensor(data=[[0.5]], grad=[[1.]], op=sum, name=sum((y + (-((x @ W) + b)))))
visited Tensor(data=[[0.3 0.2]], grad=[[1. 1.]], op=+, name=(y + (-((x @ W) + b))))
visited Tensor(data=[[-0.9 -1.1]], grad=[[1. 1.]], op=neg, name=(-((x @ W) + b)))
visited Tensor(data=[[0.9 1.1]], grad=[[-1. -1.]], op=+, name=((x @ W) + b))
visited Tensor(data=[[0.8 0.9]], grad=[[-1. -1.]], op=@, name=(x @ W))
visited Tensor(data=[[1. 0. 1.]], grad=[[-0.8 -0.7 -0.9]], op=, name=x)
visited Tensor(data=[[0.6 0.2]
 [0.3 0.4]
 [0.2 0.7]], grad=[[-1. -1.]
 [ 0.  0.]
 [-1. -1.]], op=, name=W)
visited Tensor(data=[[0.1 0.2]], grad=[[-1. -1.]], op=, name=b)
visited Tensor(data=[[1.2 1.3]], grad=[[1. 1.]], op=, name=y)
```

The first tensor is exactly our loss, which has a gradient of 1 w.r.t itself. It was formed by summing across the given tensor given by `y-y_hat`. The next step is to get `y-y_hat`, and the flow continues. 