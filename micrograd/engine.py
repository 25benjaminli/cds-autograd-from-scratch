
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    # each of these operations silently builds the computational graph by adding new values 
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+') # therefore, the previous two nodes are self and other

        # each operation contains a _backward function that updates the gradients of the inputs
        # distribute the CURRENT gradient to its inputs (going backwards)
        # e.g. out = self + other, then d(out)/d(self) = 1, d(out)/d(other) = 1.
        # dL/d(self) = dL/d(out) * d(out) / dself = out.grad * 1 = out.grad
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # out = self * other
        # dL/d(self) = dL/d(out) * d(out)/d(self) = out.grad * other.data
        # dL/d(other) = dL/d(out) * d(out)/d(other) = out.grad * self.data
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        # out = self^other
        # dL/d(other) = dL/d(out) * d(out)/d(other) = out.grad * other * self.data^(other-1)
        # just the power rule lol
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        # out = max(0, x)
        # dL/d(self) = dL/d(out) * d(out)/d(self) = out.grad * (0 if data < 0, otherwise 1)
        def _backward():
            self.grad += (out.data > 0) * out.grad
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
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"