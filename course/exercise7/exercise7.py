"""
Exercise 7: Adam versus plain gradient descent.

With H = 20 on the training set, trains for 20 000 epochs with SGD at
lr = 0.01, 0.1, 0.3, 1.0 and with Adam at lr = 0.01, 0.001, plots all six
training-loss curves on one set of log-log axes, and reports the first epoch
at which the training loss drops below 0.2 (plus final train/val loss for (c)).
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

H, EPOCHS, THRESHOLD = 20, 20000, 0.2
runs = [("sgd", 0.01), ("sgd", 0.1), ("sgd", 0.3), ("sgd", 1.0),
        ("adam", 0.01), ("adam", 0.001)]
epochs = np.arange(1, EPOCHS + 1)

fig, ax = plt.subplots(figsize=(7, 5))
print(f"{'optimiser':>9} {'lr':>6} {'first epoch < 0.2':>18} "
      f"{'final train':>12} {'final val':>10}")
for opt, lr in runs:
    net, history = fit.train(xtr, ytr, hidden=H, epochs=EPOCHS, lr=lr, optimiser=opt)
    train = np.array(history["train"])
    below = np.flatnonzero(train < THRESHOLD)     # NaN compares False, so divergence counts as never
    first = str(below[0] + 1) if below.size else "never"
    print(f"{opt:>9} {lr:6g} {first:>18} {train[-1]:12.4f} "
          f"{fit.mse(net, xva, yva):10.4f}")
    ax.loglog(epochs, train, label=f"{opt}, lr = {lr:g}")
ax.axhline(THRESHOLD, color="k", ls=":", lw=1, label="loss = 0.2")
ax.set_xlabel("epoch")
ax.set_ylabel("training loss (MSE)")
ax.set_title(f"H = {H}: SGD vs Adam")
ax.legend()
plt.tight_layout()
plt.savefig(HERE / "exercise7.png", dpi=120)
