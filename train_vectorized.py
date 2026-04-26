"""
general structure:
1. Load MNIST
2. Do the train/test split
3. Initialize MLP
4. For each epoch: you'll pass data through MLP, compute loss w.r.t ground truth, then backprop and update gradients
5. Evaluate on test set
"""

import numpy as np
from vectorized_nn import MLP as MLP_vec
from vectorized_engine import Tensor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt

# silence runtime errors, not the best practice but it's working
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    # use a generator to yield batches
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]

def run():
    np.random.seed(42) # reproducibility
    
    """hyperparameters"""
    lr = 0.05
    batch_size = 32
    X, y = load_digits(return_X_y=True)
    X = X / 16.0 # normalize pixel values to [0, 1] since they're originally in [0, 16]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    in_features = X_train.shape[1] # 64 for 8x8 images
    out_features = len(np.unique(y_train))
    
    model = MLP_vec(in_features, [128, 64, out_features])

    # print("X, y shapes:", X_train_tensor.data.shape, y_train_tensor.data.shape)

    print("number of training samples", X_train.shape[0])
    t1 = time.time()
    for epoch in range(10):
        mean_loss = 0
        # shuffle both before training
        X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
        batches = get_batched_data(X_train, y_train, batch_size)
        for sample_x, sample_y_idx in batches:
            sample_y = np.eye(out_features)[sample_y_idx]
            x_tensor = Tensor(sample_x, _name="sample x")
            y_tensor = Tensor(sample_y, _name="sample y")
            logits = model(x_tensor)
            loss = cross_entropy(logits.softmax(), y_tensor)
            loss.backward()
            
            # very simply update params w/ SGD
            scale = 1.0 / len(sample_x)
            for p in model.parameters():
                p.grad *= scale
                p.data -= lr * p.grad
            
            model.zero_grad()
            mean_loss += loss.data.item()
        mean_loss /= len(X_train)
        print(f"epoch {epoch}, mean loss: {mean_loss:.4f}")

    print("total training time", time.time() - t1)
    # eval on test set
    X_test_tensor = Tensor(X_test, _name="X_test")
    y_test_tensor = Tensor(np.eye(out_features)[y_test], _name="Y_test")

    preds = model(X_test_tensor).softmax()
    print("accuracy", accuracy(preds, y_test_tensor))

    # visualize four random examples, for each retrieve top 3 predictions and the softmax values
    rand_indices = np.random.randint(0,len(X_test),size=4)
    rand_X, rand_y = X_test[rand_indices], y_test[rand_indices]
    test_preds = model(Tensor(rand_X)).softmax().data

    print("X, y shapes:", rand_X.shape, rand_y.shape, test_preds.shape)

    fig, axes = plt.subplots(2,2, figsize=(10, 7))

    rand_X = rand_X.reshape(-1, 8, 8) # reshape to 8x8 for vis
    
    for idx, ax in enumerate(axes.flatten()):
        ax.axis("off")
        ax.imshow(rand_X[idx])
        top3 = np.argsort(test_preds[idx])[::-1][:3]
        conf = test_preds[idx][top3]
        ax.set_title(
			f"number {rand_y[idx]} top 3 {top3} conf {np.round(conf, 2)}",
			fontsize=9,
		)

    fig.tight_layout(pad=1.2)
    plt.show()


if __name__ == "__main__":
    run()