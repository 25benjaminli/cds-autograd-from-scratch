## Autograd for Deep Learning from Scratch

The goal: demystify PyTorch's `loss.backward()` by building an autograd engine from scratch. Implements vectorized operations, optimizers, as well as common layers / activation functions. Other features:
- Graph visualization for the backward pass
- Unit tests with numerical differentiation to verify validity
- Evaluate common architectures (e.g. MLP, CNN, or even a transformer) using said autograd engine
- (time permitting) Use cupy for gpu-accelerated operations

Useful repositories:
- https://github.com/karpathy/micrograd: simple implementation of basic ops but not vectorized
- https://github.com/eduardoleao052/Autograd-from-scratch: more comprehensive implementation but lacks visualizations, tests, gpu accelerated ops

Good article to learn more about autograd: https://medium.com/sfu-cspmp/diy-deep-learning-crafting-your-own-autograd-engine-from-scratch-for-effortless-backpropagation-ddab167faaf5

Overall idea:
- In the forwards pass, compute a DAG with all the operations + values
- In the backwards pass, compute a topological ordering of said graph and traverse, computing the partial derivatives w.r.t loss along the way