import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray, _children=(), _op='', _name=None):
        self.data = np.array(data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"tensor data must be 2D, got shape {self.data.shape}")
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._name = _name

    def assert_same_shape(self, other):
        if self.data.shape != other.data.shape:
            raise ValueError(f"shape mismatch: {self.data.shape} vs {other.data.shape}")
    
    @staticmethod
    def _reduce_grad_to_shape(grad, target_shape):
        """Reduce grad from broadcasted shape back to target_shape by summing."""
        # Sum over axes where target has size 1 but grad is larger
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if target_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
    
    @staticmethod
    def _as_tensor_2d(other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if other.data.ndim != 2:
            raise ValueError(f"data must be 2D when converting to tensor, got shape {other.data.shape}")
        return other
    
    def __add__(self, other):
        """
        element wise addition
        """
        other = Tensor._as_tensor_2d(other)
        # self.assert_same_shape(other)
        a_bcast, b_bcast = np.broadcast_arrays(self.data, other.data)
        out = Tensor(a_bcast + b_bcast, (self, other), "+", _name=f"({self._name} + {other._name})")

        def _backward():
            # For Z = X + Y, dL/dX = dL/dZ * dZ/dX = out.grad * 1 = out.grad. Same for dL/dY. 
            self.grad += Tensor._reduce_grad_to_shape(out.grad, self.data.shape)
            other.grad += Tensor._reduce_grad_to_shape(out.grad, other.data.shape)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = Tensor._as_tensor_2d(other)
        try:
            a_bcast, b_bcast = np.broadcast_arrays(self.data, other.data)
        except ValueError as e:
            raise ValueError(f"mul shape mismatch: {self.data.shape} vs {other.data.shape}") from e

        out = Tensor(a_bcast * b_bcast, (self, other), "*", _name=f"({self._name} * {other._name})")
        
        def _backward():
            # for Z = X * Y, dL/dX = dL/dZ * dZ/dX = out.grad * Y, and dL/dY = out.grad * X.
            self.grad += Tensor._reduce_grad_to_shape(b_bcast * out.grad, self.data.shape)
            other.grad += Tensor._reduce_grad_to_shape(a_bcast * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = Tensor._as_tensor_2d(other)
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(f"matmul shape mismatch: {self.data.shape} @ {other.data.shape}")
        out = Tensor(self.data @ other.data, (self, other), "@", _name=f"({self._name} @ {other._name})")

        def _backward():
            # For Y = X @ W:
            # dL/dX = dL/dY @ W^T
            # dL/dW = X^T @ dL/dY
            # thankfully we don't have to compute the entire jacobian, which is higher dimensional
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    

    def sum(self):
        out = Tensor(np.array([[self.data.sum()]]), (self,), "sum", _name=f"sum({self._name})")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(self.data * (self.data > 0), (self,), "ReLU", _name=f"ReLU({self._name})")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def softmax(self):
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        out = Tensor(exps / exps.sum(axis=1, keepdims=True), 
                     (self,), "softmax", _name=f"softmax({self._name})") # (B,C)
        
        def _backward():
            # apparently this gradient is more efficient, so we'll use it
            s = out.data
            g = out.grad
            self.grad += s * (g - (g * s).sum(axis=1, keepdims=True))
        
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,), "log", _name=f"log({self._name})")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        # topological order all of the children in the graph, it does a DFS to collect the nodes in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # reverse the topo order so we go from output -> input
        if self.data.shape != (1, 1):
            raise ValueError(f"backward() expects scalar loss (shape (1, 1)), got {self.data.shape}")
        self.grad = np.ones_like(self.data) # gradient w.r.t itself is 1
        for v in reversed(topo):
            # print("visited", v)
            v._backward()

    
    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, (self,), f"pow({exponent})", _name=f"({self._name}**{exponent})")

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out
    

    def __neg__(self): # -self
        out = Tensor(-self.data, (self,), "neg", _name=f"(-{self._name})")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other): # self - other
        return self + (-other)
    
    # these r methods are called if the object is on the right side
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        other = Tensor._as_tensor_2d(other)
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, other):
        return Tensor._as_tensor_2d(other) @ self

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op}, name={self._name})"
    
if __name__ == "__main__":
    # broadcast_arrays returns arrays broadcasted to the same shape.
    # in the case of ml, you might have W @ x + b, where W @ x is (B, n) and b is (n,), so you want to broadcast b to (B, n) before adding. 
    # a = np.array([[1, 2], [3, 4], [5,6]]) # (3,2)
    # b = np.array([0.1,0.2]) # (2,)
    # print("a + b:", a + b)
    x = Tensor(np.array([[1.0, 0.0, 1.0]]))          # (1,3)
    W = Tensor(np.array([[1.0, 0.0],                  # (3,2)
                        [0.0, 1.0],
                        [1.0, 0.0]]))
    b = Tensor(np.array([[0.0, 0.0]]))                # (1,2)
    y = Tensor(np.array([[3.0, 1.0]]))                # (1,2)

    y_hat = x @ W + b   # [1+0+1, 0+0+0] = [2, 0]
    loss = ((y - y_hat) ** 2).sum()   # (3-2)^2 + (1-0)^2 = 2.0
    loss.backward()
    print("W grad:", W.grad) # should be [[-2, -2], [0, 0], [-2, -2]]
