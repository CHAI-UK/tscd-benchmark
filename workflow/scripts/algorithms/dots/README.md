# DOTS

DOTS is our diffusion-based causal discovery method for time series, introduced in the paper accompanying this benchmark (see the repository README for the citation).

DOTS builds on the [DiffAN](../diffan) implementation (Sanchez et al., 2023): it reuses DiffAN's score-model training, topological ordering and CAM pruning, and extends them to lagged time series data by

1. computing **multiple topological orderings** across diffusion steps (`n_ord` hyperparameter),
2. aggregating them via **soft voting** with threshold `theta`, and
3. enforcing **temporal constraints** on the resulting window graph (`constraint`, `const_inst` hyperparameters).

The base DiffAN implementation lives in the sibling [`diffan/`](../diffan) directory and is treated as a separate, competing method in the benchmark. Only the code in this directory is specific to DOTS.

## Files

- `dots.py` — the `DOTS` model (subclass of `DiffAN`) plus the ordering-aggregation (soft voting) helpers.
- `main.py` — Snakemake entry point (referenced from the benchmark configs as `dots/main.py`).
- `demo.ipynb` — minimal standalone demo comparing DOTS with base DiffAN on the bundled `test_data/`.
- `ablation_orders.py` / `ablation_orders.ipynb` — standalone scripts for the orderings ablation reported in the paper; results are accumulated in `test_data/ablation_orders.csv` and plotted by `analysis/dots_ablation_orders.ipynb`.
- `test_data/` — a small synthetic dataset used by the demo and the orderings ablation.

## Environment

DOTS runs in the same conda environment as DiffAN (`workflow/envs/diffan.yml`), since it shares the diffusion machinery and the R-based CAM pruning.
