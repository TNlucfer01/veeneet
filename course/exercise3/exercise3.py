"""
Exercise 3: early stopping.

Trains H = 20 for 20 000 epochs with the validation set recorded, plots the
training and validation loss against epoch on log-log axes, finds the epoch
with the smallest validation loss, then retrains for just that many epochs
and plots the final and early-stopped curves over the training data.
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
xgrid = np.linspace(-2, 2, 400)

H, EPOCHS = 20, 20000
net, history = fit.train(xtr, ytr, hidden=H, epochs=EPOCHS, xval=xva, yval=yva)
train, val = np.array(history["train"]), np.array(history["val"])
epochs = np.arange(1, EPOCHS + 1)          # history[i] is the loss after epoch i+1

# (a)
best = int(np.argmin(val))
best_epoch = best + 1
print(f"smallest validation loss at epoch {best_epoch}")
print(f"  at epoch {best_epoch:5d}: train = {train[best]:.4f}  val = {val[best]:.4f}")
print(f"  at epoch {EPOCHS:5d}: train = {train[-1]:.4f}  val = {val[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(epochs, train, label="training loss")
ax.loglog(epochs, val, label="validation loss")
ax.axvline(best_epoch, color="k", ls="--", lw=1,
           label=f"min validation loss (epoch {best_epoch})")
ax.set_xlabel("epoch")
ax.set_ylabel("mean squared error")
ax.set_title(f"H = {H}: loss during training")
ax.legend()
plt.tight_layout()
plt.savefig(HERE / "exercise3_loss.png", dpi=120)

# (c) retrain for just best_epoch epochs; same seed, so it follows the same path
net_es, history_es = fit.train(xtr, ytr, hidden=H, epochs=best_epoch)
print(f"early-stopped network: train = {fit.mse(net_es, xtr, ytr):.4f}  "
      f"val = {fit.mse(net_es, xva, yva):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, n, title in [(axes[0], net, f"end of training ({EPOCHS} epochs)"),
                     (axes[1], net_es, f"early stopped ({best_epoch} epochs)")]:
    ax.plot(xtr, ytr, "o", ms=5, label="training data")
    ax.plot(xgrid, fit.predict(n, xgrid), "r", label="fitted curve")
    ax.set_ylim(-2.5, 3)
    ax.set_xlabel("x")
    ax.set_title(f"H = {H}, {title}")
    ax.legend()
axes[0].set_ylabel("y")
plt.tight_layout()
plt.savefig(HERE / "exercise3_curves.png", dpi=120)
