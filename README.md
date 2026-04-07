## Autograd for Deep Learning from Scratch

The goal: demystify PyTorch's `loss.backward()` by building an autograd engine from scratch. Implements vectorized operations, optimizers, common layers / activation functions, and validates on real world tasks. 

Useful repositories:
- https://github.com/karpathy/micrograd: simple implementation of basic ops but not vectorized
- https://github.com/eduardoleao052/Autograd-from-scratch: more comprehensive implementation but lacks visualizations, tests, gpu accelerated ops

Good articles for math foundations:
- https://robotchinwag.com/posts/linear-layer-deriving-the-gradient-for-the-backward-pass/
- https://cs231n.stanford.edu/handouts/linear-backprop.pdf

Also check out [example.md](example.md) for a sample problem, which is also implemented in `vectorized_engine.py`. 

Todos:
- [x] implement vectorized engine for 2D matrices
- [x] add explanation (example.md)
- [ ] add broadcasting
- [ ] support higher dimensional (n>=3) tensor computations
- [ ] add optimizers
- [ ] test MLP / CNN / transformer on datasets
- [ ] unit tests with numerical differentiation to verify validity
- [ ] graph visualizations? 
- [ ] use cupy for gpu-accelerated operations? 