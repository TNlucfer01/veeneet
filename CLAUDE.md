# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for SEMTM0051 Foundations of AI: "underfitting, overfitting and held-out data" (brief in `coursework.pdf`). The work is experiments that fit small 1-D regression networks to simulated data and compare training, validation and test error. The `uv` package scaffold (`src/veenth/`) is empty. All real code is in `course/`.

## Environment and commands

Managed with `uv` (Python 3.14, deps: torch and matplotlib; numpy comes in through them). Run scripts **from inside `course/`**: they `import fit` as a sibling module and read `data.csv` by relative path.

```sh
uv sync                                  # create .venv and install deps
cd course
uv run python make_data.py               # regenerate data.csv (120 rows, SEED=2026)
uv run python exersise1.py               # worked example; writes exercise1.png
```

There is no test suite, linter or build step. `course/test.py` is a scratch file, not tests.

## Architecture

- `course/fit.py` is the shared library that every exercise script imports. It holds `load`, `split` (seeded shuffle into train/val/test, with test being the leftover), `make_net(H)` (one sigmoid hidden layer with 3H+1 params, where `H=0` gives a plain linear model), `train` (full-batch Adam or SGD, returns `(net, history)` where `history['train']` and optionally `history['val']` hold the per-epoch MSE), `mse` and `predict`. It sets `torch.set_default_dtype(torch.float64)` globally. The brief says the basic exercises should use `fit.py` unchanged, and only the bonus questions change its arguments or behaviour. Keep its public API stable.
- `course/make_data.py` generates `data.csv` from a hidden `true_function` plus Gaussian noise. The brief asks students to treat the true function and noise level as unknown, and to estimate the noise floor from experiments. Don't use or reveal those values in exercise code or write-up reasoning unless the user asks.
- `course/exersise1.py` (the filename is misspelled) is the provided template. Later exercises are meant to copy and adapt it: loop over hidden sizes or splits, call `fit.train(..., epochs=20000)`, plot with matplotlib and save a PNG next to the script.

## Conventions

- Reproducibility comes from explicit seeds (`split(seed=0)` and `train(seed=0)` by default). Keep seeds fixed when comparing configurations.
- According to the brief, a single training run should take at most about 20 s on a laptop CPU. If a run takes much longer, that points to a mistake.
- The user annotates `fit.py` with their own learning comments and questions (e.g. "why do we have this"). Leave those comments alone.
