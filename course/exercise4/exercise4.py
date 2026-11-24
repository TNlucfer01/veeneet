"""
Exercise 4: weight decay.

With H = 20 and 20 000 epochs, trains with lambda = 0, 1e-4, 1e-3, 1e-2, 1e-1,
tabulates the training and validation losses, and plots the fitted curves for
lambda = 0, the best lambda (smallest validation loss) and lambda = 1e-1.
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
lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

nets, val_loss = {}, []
print(f"{'lambda':>8} {'train MSE':>10} {'val MSE':>10}")
for lam in lambdas:
    net, history = fit.train(xtr, ytr, hidden=H, epochs=EPOCHS, weight_decay=lam)
    nets[lam] = net
    val_loss.append(fit.mse(net, xva, yva))
    print(f"{lam:8.0e} {history['train'][-1]:10.4f} {val_loss[-1]:10.4f}")
best = lambdas[int(np.argmin(val_loss))]
print(f"smallest validation loss at lambda = {best:g}")

shown = [0.0, best, 1e-1] if best not in (0.0, 1e-1) else [0.0, 1e-1]
fig, axes = plt.subplots(1, len(shown), figsize=(4 * len(shown), 4), sharey=True)
for ax, lam in zip(axes, shown):
    ax.plot(xtr, ytr, "o", ms=5, label="training data")
    ax.plot(xgrid, fit.predict(nets[lam], xgrid), "r", label="fitted curve")
    ax.set_ylim(-2.5, 3)
    ax.set_xlabel("x")
    tag = " (best)" if lam == best else ""
    ax.set_title(f"H = {H}, lambda = {lam:g}{tag}")
    ax.legend()
axes[0].set_ylabel("y")
plt.tight_layout()
plt.savefig(HERE / "exercise4.png", dpi=120)
