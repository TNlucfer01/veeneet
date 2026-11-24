"""
Exercise 2: choosing capacity with a validation set.

Trains H = 0, 1, 2, 3, 5, 10, 20, 50, 100 on the training set only for
20 000 epochs, records the final training loss and the validation loss, and
plots both against H on a log axis (H = 0 drawn at 0.5). Done for seed=0 and,
for part (d), again with seed=1.
"""

import sys
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit


x, y = fit.load(HERE.parent / "data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)


sizes = [0, 1, 2, 3, 5, 10, 20, 50, 100]
hplot = [0.5 if H == 0 else H for H in sizes]   # H = 0 plotted at 0.5
seed=[0,1]  # for part (d) we will repeat the experiment with a different seed
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, seed in zip(axes, seed):
    train_loss, val_loss = [], []
    print(f"seed = {seed}")
    print(f"{'H':>5} {'params':>7} {'train MSE':>10} {'val MSE':>10}")
    for H in sizes:
        
        net, history = fit.train(xtr, ytr, hidden=H, epochs=20000, seed=seed)
        train_loss.append(history["train"][-1])
        val_loss.append(fit.mse(net, xva, yva))
        print(f"{H:5d} {fit.n_params(net):7d} {train_loss[-1]:10.4f} "
              f"{val_loss[-1]:10.4f}")
    best = sizes[int(np.argmin(val_loss))]
    print(f"smallest validation loss at H = {best}\n")
    ax.plot(hplot, train_loss, "o-", label="training loss")
    ax.plot(hplot, val_loss, "s-", label="validation loss")
    ax.set_xscale("log")
    ax.set_xticks(hplot)
    ax.set_xticklabels(["0"] + [str(H) for H in sizes[1:]])
    ax.set_xlabel("number of hidden units H")
    ax.set_title(f"seed = {seed}")
    ax.legend()
axes[0].set_ylabel("mean squared error")
plt.tight_layout()
plt.savefig(HERE / "exercise2.png", dpi=120)