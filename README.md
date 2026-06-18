# Pedestrian-flow simulator reimplementation

Python 3.10 code for the 1D modified social-force / hard-body pedestrian model from Seyfried, Steffen & Lippert (2006).

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install numpy pandas matplotlib
```

## Run

Fast check:

```bash
python run_experiments.py --quick
```

Paper-scale run, using 300,000 relaxation steps and 300,000 measurement steps:

```bash
python run_experiments.py --full
```

Outputs are saved in `output/`:

- `quick_results.csv` or `full_results.csv`
- velocity-density diagram PNG
- position-history CSV/PNG for density-wave examples near rho = 1.16 and 1.21

## Implemented model

- Periodic 1D corridor with `L = 17.3 m`.
- Intended speeds sampled from `Normal(1.24, 0.05)`.
- Driving term `(v0 - v) / tau`, with `tau = 0.61 s`.
- Required length `d = a + b v`, with `a = 0.36 m` and tested `b = 0, 0.56, 1.06 s`.
- Hard-body interaction without remote action.
- Hard-body interaction with remote action using `e = 0.07`, `f = 2`.
