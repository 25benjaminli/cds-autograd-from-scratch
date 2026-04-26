"""
general structure:
1. Load MNIST
2. Do the train/test split
3. Initialize MLP
4. For each epoch: you'll pass data through MLP, compute loss w.r.t ground truth, then backprop and update gradients
5. Evaluate on test set

TODOs:
[x] get our vectorized implementation working
[] compare speed / other metrics vs. karpathy's micrograd
"""

import numpy as np
from vectorized_nn import MLP as MLP_vec
from micrograd_nn import MLP as MLP_micro
from vectorized_engine import Tensor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import time

def cross_entropy(y_pred, y_true):
    batch_size = y_true.data.shape[0]
    # epsilon to avoid zeroes in log
    eps = Tensor(np.full_like(y_pred.data, 1e-12), _name="eps")
    # hacky way to do mean reduction
    scale = Tensor(np.array([[1.0 / batch_size]]), _name="inv_batch")
    return -(y_true * (y_pred + eps).log()).sum() * scale

def accuracy(y_pred, y_true):
    # argmax because y_pred and y_true are one-hot encoded, reducing along class dimension
    pred_labels = np.argmax(y_pred.data, axis=1)
    true_labels = np.argmax(y_true.data, axis=1)
    return (pred_labels == true_labels).mean()

def get_batched_data(X, y, batch_size):
    X, y = shuffle(X, y)
    X_batches = [
        X[i:i+batch_size] if batch_size > 1 else 
        np.expand_dims(X[i:i+batch_size], axis=0) 
        for i in range(0, X.shape[0], batch_size)
    ]
    y_batches = [
        y[i:i+batch_size] if batch_size > 1
        else np.expand_dims(y[i:i+batch_size], axis=0)
        for i in range(0, y.shape[0], batch_size)
    ]
    return X_batches, y_batches

def run():
    np.random.seed(42) # reproducibility
    
    """hyperparameters"""
    lr = 0.05
    batch_size = 32
    use_vectorized = False # right now this doesn't work, we need to test it

    X, y = load_digits(return_X_y=True)
    X = X / 16.0 # normalize pixel values to [0, 1] since they're originally in [0, 16]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    in_features = X_train.shape[1] # 64 for 8x8 images
    out_features = len(np.unique(y_train))
    
    if use_vectorized:
        model = MLP_vec(in_features, [128, 64, out_features])
    else:
        model = MLP_micro(in_features, [128, 64, out_features])

    # one-hot encode the labels, np.eye creates the identity matrix and we just select appropriate indices
    y_train_one_hot = np.eye(out_features)[y_train]
    # print("X, y shapes:", X_train_tensor.data.shape, y_train_tensor.data.shape)

    print("number of training samples", X_train.shape[0])
    t1 = time.time()
    for epoch in range(10):
        mean_loss = 0
        # shuffle both before training
        X_train_batches, y_train_batches = get_batched_data(X_train, y_train_one_hot, batch_size)
        for sample_x, sample_y in zip(X_train_batches, y_train_batches):
            x_tensor = Tensor(sample_x, _name="sample x")
            y_tensor = Tensor(sample_y, _name="sample y")
            logits = model(x_tensor)
            loss = cross_entropy(logits.softmax(), y_tensor)
            loss.backward()
            
            # very simply update params w/ SGD
            for p in model.parameters():
                p.data -= lr * p.grad
            
            model.zero_grad()
            mean_loss += loss.data.item()
        mean_loss /= len(X_train_batches)
        print(f"epoch {epoch}, mean loss: {mean_loss:.4f}")

    print("total training time", time.time() - t1)
    # eval on test set
    X_test_tensor = Tensor(X_test, _name="X_test")
    y_test_tensor = Tensor(np.eye(out_features)[y_test], _name="Y_test")

    preds = model(X_test_tensor).softmax()
    print("accuracy", accuracy(preds, y_test_tensor))

    # TODO: visualize a few examples


if __name__ == "__main__":
    run()