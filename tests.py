import numpy as np

from vectorized_engine import Tensor

# element wise add tests

def test_add_forward_basic():
    # Make sure a forward pass produces corect values
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), _name="a")
    b = Tensor(np.array([[10.0, 20.0], [30.0, 40.0]]), _name="b")
    c = a + b
    expected = np.array([[11.0, 22.0], [33.0, 44.0]])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"
    print("Passed 'test_add_forward_basic'")

def test_add_backward_gradients():
    # Backward pass correctly accumulates grad=1 for both inputs during a sum loss.
    a = Tensor(np.array([[1.0, 2.0]]), _name="a")
    b = Tensor(np.array([[3.0, 4.0]]), _name="b")
    c = a + b
    loss = c.sum()
    loss.backward()
    assert np.allclose(a.grad, np.ones_like(a.data)), f"a.grad wrong: {a.grad}"
    assert np.allclose(b.grad, np.ones_like(b.data)), f"b.grad wrong: {b.grad}"
    print("Passed 'test_add_backward_gradients'")

def test_add_backward_chain():
    # Gradients flow correctly through a chain
    a = Tensor(np.array([[1.0, 2.0]]), _name="a")
    b = Tensor(np.array([[3.0, 4.0]]), _name="b")
    c = Tensor(np.array([[2.0, 5.0]]), _name="c")
    loss = ((a + b) * c).sum()
    loss.backward()
    assert np.allclose(a.grad, c.data), f"a.grad wrong: {a.grad}"
    assert np.allclose(b.grad, c.data), f"b.grad wrong: {b.grad}"
    print("Passed test_add_bacckward_chain")

def test_add_broadcast_row_bias():
    # adding (1, N) bias to a (B, N) tensor broadcasts correctly both forward and backward
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), _name="a")  
    b = Tensor(np.array([[10.0, 20.0]]), _name="b")           
    c = a + b
    expected = np.array([[11.0, 22.0], [13.0, 24.0], [15.0, 26.0]])
    assert np.allclose(c.data, expected), f"Forward wrong: {c.data}"
    loss = c.sum()
    loss.backward()
    assert np.allclose(a.grad, np.ones((3, 2))), f"a.grad wrong: {a.grad}"
    assert np.allclose(b.grad, np.array([[3.0, 3.0]])), f"b.grad wrong: {b.grad}"
    print("Passed 'test_add_broadcast_row_bias'")

def test_add_gradient_accumulation():
    # Using the same tensor twice accumulates gradients correctly.
    a = Tensor(np.array([[2.0, 3.0]]), _name="a")
    loss = (a + a).sum()
    loss.backward()
    assert np.allclose(a.grad, 2 * np.ones_like(a.data)), f"a.grad wrong: {a.grad}"
    print("Passed 'test_add_gradient_accumulation'")

def test_add_numerical_gradient():
    # Numerical gradient check vs analytical gradient for a + b.
    np.random.seed(0)
    a_data = np.random.randn(2, 3)
    b_data = np.random.randn(2, 3)
    eps = 1e-5
 
    def forward(a_data, b_data):
        a = Tensor(a_data, _name="a")
        b = Tensor(b_data, _name="b")
        return (a + b).sum().data[0, 0]
 
    a = Tensor(a_data.copy(), _name="a")
    b = Tensor(b_data.copy(), _name="b")
    (a + b).sum().backward()
 
    for i in range(a_data.shape[0]):
        for j in range(a_data.shape[1]):
            a_plus = a_data.copy(); a_plus[i, j] += eps
            a_minus = a_data.copy(); a_minus[i, j] -= eps
            num_grad = (forward(a_plus, b_data) - forward(a_minus, b_data)) / (2 * eps)
            assert abs(a.grad[i, j] - num_grad) < 1e-6, \
                f"Numerical grad mismatch at a[{i},{j}]: analytical={a.grad[i,j]:.6f}, numerical={num_grad:.6f}"
 
    print("Passed 'test_add_numerical_gradient'")


# element wise multiplication tests

def test_mul_forward_basic():
    # Make sure a forward pass produces correct values
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), _name="a")
    b = Tensor(np.array([[10.0, 20.0], [30.0, 40.0]]), _name="b")
    c = a * b
    expected = np.array([[10.0, 40.0], [90.0, 160.0]])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"
    print("Passed 'test_mul_forward_basic'")

def test_mul_backward_gradients():
    # Backward pass correctly uses the other input during a sum loss.
    a = Tensor(np.array([[1.0, 2.0]]), _name="a")
    b = Tensor(np.array([[3.0, 4.0]]), _name="b")
    c = a * b
    loss = c.sum()
    loss.backward()
    assert np.allclose(a.grad, b.data), f"a.grad wrong: {a.grad}"
    assert np.allclose(b.grad, a.data), f"b.grad wrong: {b.grad}"
    print("Passed 'test_mul_backward_gradients'")

def test_mul_backward_chain():
    # Gradients flow correctly through a chain
    a = Tensor(np.array([[1.0, 2.0]]), _name="a")
    b = Tensor(np.array([[3.0, 4.0]]), _name="b")
    c = Tensor(np.array([[2.0, 5.0]]), _name="c")
    loss = ((a * b) + c).sum()
    loss.backward()
    assert np.allclose(a.grad, b.data), f"a.grad wrong: {a.grad}"
    assert np.allclose(b.grad, a.data), f"b.grad wrong: {b.grad}"
    print("Passed 'test_mul_backward_chain'")

def test_mul_broadcast_row():
    # Multiplying a (B, N) tensor by a (1, N) tensor broadcasts correctly.
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), _name="a")
    b = Tensor(np.array([[10.0, 20.0]]), _name="b")
    c = a * b
    expected = np.array([[10.0, 40.0], [30.0, 80.0], [50.0, 120.0]])
    assert np.allclose(c.data, expected), f"Forward wrong: {c.data}"
    loss = c.sum()
    loss.backward()
    assert np.allclose(a.grad, np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])), f"a.grad wrong: {a.grad}"
    assert np.allclose(b.grad, np.array([[9.0, 12.0]])), f"b.grad wrong: {b.grad}"
    print("Passed 'test_mul_broadcast_row'")

def test_mul_gradient_accumulation():
    # Using the same tensor twice accumulates gradients correctly.
    a = Tensor(np.array([[2.0, 3.0]]), _name="a")
    loss = (a * a).sum()
    loss.backward()
    expected = 2 * a.data
    assert np.allclose(a.grad, expected), f"a.grad wrong: {a.grad}"
    print("Passed 'test_mul_gradient_accumulation'")

def test_mul_shape_mismatch_raises():
    # Non-broadcastable shapes should raise a ValueError.
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), _name="a")
    b = Tensor(np.array([[5.0, 6.0, 7.0]]), _name="b")
    try:
        _ = a * b
        assert False, "Expected ValueError but none was raised"
    except ValueError:
        pass
    print("Passed 'test_mul_shape_mismatch_raises'")

def test_mul_numerical_gradient():
    # Numerical gradient check vs analytical gradient for a * b.
    np.random.seed(0)
    a_data = np.random.randn(2, 3)
    b_data = np.random.randn(2, 3)
    eps = 1e-5

    def forward(a_data, b_data):
        a = Tensor(a_data, _name="a")
        b = Tensor(b_data, _name="b")
        return (a * b).sum().data[0, 0]

    a = Tensor(a_data.copy(), _name="a")
    b = Tensor(b_data.copy(), _name="b")
    (a * b).sum().backward()

    for i in range(a_data.shape[0]):
        for j in range(a_data.shape[1]):
            a_plus = a_data.copy(); a_plus[i, j] += eps
            a_minus = a_data.copy(); a_minus[i, j] -= eps
            num_grad = (forward(a_plus, b_data) - forward(a_minus, b_data)) / (2 * eps)
            assert abs(a.grad[i, j] - num_grad) < 1e-6, \
                f"Numerical grad mismatch at a[{i},{j}]: analytical={a.grad[i,j]:.6f}, numerical={num_grad:.6f}"

    for i in range(b_data.shape[0]):
        for j in range(b_data.shape[1]):
            b_plus = b_data.copy(); b_plus[i, j] += eps
            b_minus = b_data.copy(); b_minus[i, j] -= eps
            num_grad = (forward(a_data, b_plus) - forward(a_data, b_minus)) / (2 * eps)
            assert abs(b.grad[i, j] - num_grad) < 1e-6, \
                f"Numerical grad mismatch at b[{i},{j}]: analytical={b.grad[i,j]:.6f}, numerical={num_grad:.6f}"

    print("Passed 'test_mul_numerical_gradient'")

# matrix multiplication tests

def test_matmul_forward_basic():
    # Forward pass produces correct values for a simple matrix multiplication
    X = Tensor(np.array([[1.0, 2.0, 3.0]]), _name="X")           # (1,3)
    W = Tensor(np.array([[1.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 1.0]]), _name="W")                 # (3,2)
    out = X @ W
    expected = np.array([[4.0, 5.0]])
    assert out.data.shape == (1, 2), f"Wrong shape: {out.data.shape}"
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    print("Passed 'test_matmul_forward_basic'")

def test_matmul_forward_batched():
    #Forward pass works for a batched input (B, in) @ (in, out).
    X = Tensor(np.array([[1.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 1.0]]), _name="X")                 # (3,2)
    W = Tensor(np.array([[2.0, 0.0, 1.0],
                         [0.0, 3.0, 1.0]]), _name="W")            # (2,3)
    out = X @ W
    expected = np.array([[2.0, 0.0, 1.0],
                         [0.0, 3.0, 1.0],
                         [2.0, 3.0, 2.0]])
    assert out.data.shape == (3, 3), f"Wrong shape: {out.data.shape}"
    assert np.allclose(out.data, expected), f"Expected {expected}, got {out.data}"
    print("Passed 'test_matmul_forward_batched'")

def test_matmul_shape_mismatch_raises():
    #Incompatible inner dimensions should raise a ValueError.
    A = Tensor(np.ones((2, 3)), _name="A")
    B = Tensor(np.ones((4, 2)), _name="B")
    try:
        _ = A @ B
        assert False, "Expected ValueError but none was raised"
    except ValueError:
        pass
    print("Passed 'test_matmul_shape_mismatch_raises'")

def test_matmul_backward_dX():
    # dL/dX = dL/dY @ W^T is computed correctly.
    X = Tensor(np.array([[1.0, 2.0, 3.0]]), _name="X")           
    W = Tensor(np.array([[1.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 1.0]]), _name="W")               
    loss = (X @ W).sum()
    loss.backward()
    expected_X_grad = np.array([[1.0, 1.0, 2.0]])
    assert np.allclose(X.grad, expected_X_grad), f"X.grad wrong: {X.grad} expected {expected_X_grad}"
    print("Passed test_matmul_backward_dX")

def test_matmul_backward_dW():
    #dL/dW = X^T @ dL/dY is computed correctly.
    X = Tensor(np.array([[1.0, 2.0, 3.0]]), _name="X")           # (1,3)
    W = Tensor(np.array([[1.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 1.0]]), _name="W")                 # (3,2)
    loss = (X @ W).sum()
    loss.backward()
    expected_W_grad = np.array([[1.0, 1.0],
                                [2.0, 2.0],
                                [3.0, 3.0]])
    assert np.allclose(W.grad, expected_W_grad), f"W.grad wrong: {W.grad}"
    print("Passed 'test_matmul_backward_dW'")

def test_matmul_gradient_accumulation():
    #Using the same weight matrix twice accumulates gradients.
    W = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), _name="W")
    X1 = Tensor(np.array([[1.0, 0.0]]), _name="X1")
    X2 = Tensor(np.array([[0.0, 1.0]]), _name="X2")
    loss = (X1 @ W + X2 @ W).sum()
    loss.backward()
    # dL/dW = X1^T @ ones + X2^T @ ones = (X1 + X2)^T @ ones
    expected = (X1.data + X2.data).T @ np.ones((1, 2))
    assert np.allclose(W.grad, expected), f"W.grad wrong: {W.grad}"
    print("Passed 'test_matmul_gradient_accumulation'")

def test_matmul_numerical_gradient():
    #Numerical gradient check for both X and W in Y = X @ W."""
    np.random.seed(42)
    X_data = np.random.randn(3, 4)
    W_data = np.random.randn(4, 2)
    eps = 1e-5
 
    def forward(X_data, W_data):
        X = Tensor(X_data, _name="X")
        W = Tensor(W_data, _name="W")
        return (X @ W).sum().data[0, 0]
 
    X = Tensor(X_data.copy(), _name="X")
    W = Tensor(W_data.copy(), _name="W")
    (X @ W).sum().backward()
 
    for i in range(X_data.shape[0]):
        for j in range(X_data.shape[1]):
            Xp = X_data.copy(); Xp[i, j] += eps
            Xm = X_data.copy(); Xm[i, j] -= eps
            num = (forward(Xp, W_data) - forward(Xm, W_data)) / (2 * eps)
            assert abs(X.grad[i, j] - num) < 1e-6, \
                f"X numerical grad mismatch at [{i},{j}]: analytical={X.grad[i,j]:.6f}, numerical={num:.6f}"
 
    for i in range(W_data.shape[0]):
        for j in range(W_data.shape[1]):
            Wp = W_data.copy(); Wp[i, j] += eps
            Wm = W_data.copy(); Wm[i, j] -= eps
            num = (forward(X_data, Wp) - forward(X_data, Wm)) / (2 * eps)
            assert abs(W.grad[i, j] - num) < 1e-6, \
                f"W numerical grad mismatch at [{i},{j}]: analytical={W.grad[i,j]:.6f}, numerical={num:.6f}"
 
    print("Passed 'test_matmul_numerical_gradient'")


# neural network layer tests1``

if __name__ == "__main__":
    test_add_forward_basic()
    test_add_backward_gradients()
    test_add_backward_chain()
    test_add_broadcast_row_bias()
    test_add_gradient_accumulation()
    test_add_numerical_gradient()

    test_mul_forward_basic()
    test_mul_backward_gradients()
    test_mul_backward_chain()
    test_mul_broadcast_row()
    test_mul_gradient_accumulation()
    test_mul_shape_mismatch_raises()
    test_mul_numerical_gradient()

    test_matmul_forward_basic()
    test_matmul_forward_batched()
    test_matmul_shape_mismatch_raises()
    test_matmul_backward_dX()
    test_matmul_backward_dW()
    test_matmul_gradient_accumulation()
    test_matmul_numerical_gradient()