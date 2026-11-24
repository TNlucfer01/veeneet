"""
Exercise 1: fitting all the data.

Fits networks with H = 0, 1, 2, 5, 20, 100 to ALL 120 points, plots each
fitted curve over the data and prints the number of parameters and the final
training loss for each. Same as ../exersise1.py, run from this folder.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # so that `import fit` finds ../fit.py
import fit

x, y = fit.load(HERE.parent / "data.csv")
xgrid = np.linspace(-2, 2, 400)

sizes = [0, 1, 2, 5, 20, 100]
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
for ax, H in zip(axes.ravel(), sizes):
    net, history = fit.train(x, y, hidden=H, epochs=20000)
    ax.plot(x, y, "o", ms=4)
    ax.plot(xgrid, fit.predict(net, xgrid), "r")
    ax.set_ylim(-2.5, 3)
    ax.set_title(f"H = {H}: {fit.n_params(net)} parameters, "
                 f"train MSE = {history['train'][-1]:.3f}")
   
for ax in axes[-1, :]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
plt.tight_layout()
plt.savefig(HERE / "exercise1.png", dpi=120)
