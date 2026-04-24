"""
general structure:
1. Load MNIST
2. Do the train/test split
3. Initialize MLP
4. For each epoch: you'll pass data through MLP, compute loss w.r.t ground truth, then backprop and update gradients
5. Evaluate on test set
"""