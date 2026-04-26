import random
import time
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from micrograd_engine import Value
from micrograd_nn import MLP

warnings.filterwarnings("ignore", category=RuntimeWarning)


def cross_entropy_loss(logits, y_idx):
    max_val = max(v.data for v in logits)
    shifted = [v + (-max_val) for v in logits]
    exp_vals = [v.exp() for v in shifted]
    sum_exp = sum(exp_vals[1:], exp_vals[0])
    log_sum_exp = sum_exp.log()
    return log_sum_exp + (-shifted[y_idx])


def predict(model, x_row):
    logits = model(list(x_row))
    return np.argmax([v.data for v in logits])


def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


def run():
    np.random.seed(42)
    random.seed(42)

    lr = 0.01
    epochs = 10
    batch_size = 16
    train_subset = 500

    X, y = load_digits(return_X_y=True)
    X = X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, y_train = X_train[:train_subset], y_train[:train_subset]

    model = MLP(X_train.shape[1], [32, 16, len(np.unique(y))])
    print("parameters:", len(model.parameters())) # 8x8 -> 32 -> 16 -> 10
    print("training samples:", X_train.shape[0])

    t1 = time.time()
    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
        epoch_loss = 0.0

        for batch_x, batch_y in tqdm(list(get_batches(X_train, y_train, batch_size)), desc=f"epoch {epoch}"):
            batch_loss = 0.0
            for x_row, y_idx in zip(batch_x, batch_y):
                logits = model(list(x_row))
                loss = cross_entropy_loss(logits, int(y_idx))
                loss.backward()
                batch_loss += loss.data

            scale = 1.0 / len(batch_x)
            for p in model.parameters():
                p.data -= lr * p.grad * scale
            
            model.zero_grad()

            epoch_loss += batch_loss

        print(f"epoch {epoch}, loss: {epoch_loss / len(X_train):.4f}")

    print(f"training time: {time.time() - t1:.1f}s") # ~280s
    preds = np.array([predict(model, x) for x in X_test])
    print(f"test accuracy: {(preds == y_test).mean():.4f}")


if __name__ == "__main__":
    run()