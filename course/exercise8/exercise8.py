"""
Exercise 8: test data from somewhere else (distribution shift).

Question: does the validation loss, measured on x drawn like the training
data, still predict the error -- and still pick the best H -- when the test
points come from a different x-distribution?

Networks are trained on (xtr, ytr) only, H is compared on the validation set
only, and the coursework test set (xte, yte) is not used at all. The shifted
"test" sets are fresh points from shifted_data.py:
    uniform  x ~ U[-2, 2]               (no shift, the reference)
    edges    x piled up near +-2        (shift inside the training range)
    outside  |x| in [2, 2.5]            (extrapolation)
Each run is repeated over three weight-initialisation seeds.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit
import make_data
import shifted_data

x, y = fit.load(HERE.parent / "data.csv")
(xtr, ytr), (xva, yva), _ = fit.split(x, y, n_train=30, n_val=30)

rng = np.random.default_rng(8)
tests = {name: gen(10000, rng) for name, gen in
         [("uniform", shifted_data.uniform), ("edges", shifted_data.edges),
          ("outside", shifted_data.outside)]}

sizes = [1, 2, 3, 5, 10, 20, 50]
seeds = [0, 1, 2]
EPOCHS = 20000

# err[name][i, j] = error of the network with H = sizes[i], seed = seeds[j]
err = {name: np.zeros((len(sizes), len(seeds))) for name in ["validation", *tests]}
nets = {}
for i, H in enumerate(sizes):
    for j, seed in enumerate(seeds):
        net, _ = fit.train(xtr, ytr, hidden=H, epochs=EPOCHS, seed=seed)
        nets[H, seed] = net
        err["validation"][i, j] = fit.mse(net, xva, yva)
        for name, (xt, yt) in tests.items():
            err[name][i, j] = fit.mse(net, xt, yt)

print(f"noise floor sigma^2 = {shifted_data.NOISE**2:.3f}\n")
print("mean over seeds (min - max)")
print(f"{'H':>4} " + " ".join(f"{name:>22}" for name in err))
for i, H in enumerate(sizes):
    cells = [f"{e[i].mean():8.3f} ({e[i].min():.3f}-{e[i].max():.3f})" for e in err.values()]
    print(f"{H:4d} " + " ".join(f"{c:>22}" for c in cells))

print("\nH chosen by each error (per seed)")
for name, e in err.items():
    print(f"  {name:>10}: " + ", ".join(f"seed {s}: H = {sizes[k]}"
                                       for s, k in zip(seeds, e.argmin(axis=0))))

# how well does validation loss rank the networks under each test distribution?
print("\nSpearman rank correlation between validation loss and test error")
v = err["validation"].ravel()
for name in tests:
    t = err[name].ravel()
    rv, rt = np.argsort(np.argsort(v)), np.argsort(np.argsort(t))
    print(f"  {name:>8}: {np.corrcoef(rv, rt)[0, 1]:.2f}")

# figure 1: errors against H
fig, ax = plt.subplots(figsize=(7, 5))
for name, e in err.items():
    ax.errorbar(sizes, e.mean(axis=1),
                yerr=[e.mean(axis=1) - e.min(axis=1), e.max(axis=1) - e.mean(axis=1)],
                marker="o", capsize=3, label=name)
ax.axhline(shifted_data.NOISE**2, color="k", ls=":", lw=1, label="noise floor")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels([str(H) for H in sizes])
ax.set_xlabel("number of hidden units H")
ax.set_ylabel("mean squared error (mean over 3 seeds, bars = range)")
ax.set_title("Validation loss vs error under distribution shift")
ax.legend()
plt.tight_layout()
plt.savefig(HERE / "exercise8_errors.png", dpi=120)

# figure 2: what the networks do outside [-2, 2]
xgrid = np.linspace(-2.5, 2.5, 500)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, H in zip(axes, [3, 10, 50]):
    ax.axvspan(-2.5, -2, color="0.9")
    ax.axvspan(2, 2.5, color="0.9")
    ax.plot(xtr, ytr, "o", ms=5, label="training data")
    ax.plot(xgrid, make_data.true_function(xgrid), "k--", label="true g")
    for seed in seeds:
        ax.plot(xgrid, fit.predict(nets[H, seed], xgrid), lw=1.2, label=f"seed {seed}")
    ax.set_ylim(-3, 4)
    ax.set_xlabel("x")
    ax.set_title(f"H = {H} (shaded: outside training range)")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE / "exercise8_curves.png", dpi=120)
