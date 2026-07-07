# Time Series Causal Discovery (TSCD) Benchmark

A [Snakemake](https://snakemake.readthedocs.io/)-based benchmark for **causal discovery from time series**. Given a dataset (synthetic or real) and a set of algorithms, the workflow runs every algorithm × hyperparameter × dataset-instance combination, evaluates the estimated causal graphs against the ground truth, and aggregates everything into a single results table (plus optional plots).

This repository accompanies the paper introducing **DOTS**, our diffusion-based causal discovery method for time series, and can be used both to **replicate the paper's results** and as a **general framework** for benchmarking your own methods.

**Citation:**
> Pedro Sanchez, Damian Machlanski, Steven McDonagh, Sotirios A. Tsaftaris.
> *Causal Ordering for Structure Learning from Time Series*. TMLR 2025.
> Code: https://github.com/CHAI-UK/DOTS

```
@article{sanchez2025causal,
title={Causal Ordering for Structure Learning from Time Series},
author={Pedro Sanchez and Damian Machlanski and Steven McDonagh and Sotirios A. Tsaftaris},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=hWuTzqggSd}}
```

## What's included

**Algorithms** (each entry in a config file is an algorithm variant):

| Method | Config `file` | Notes |
|---|---|---|
| **DOTS** (ours) | `dots/main.py` | Built on DiffAN; multiple orderings + soft voting + temporal constraints. See [its README](workflow/scripts/algorithms/dots/README.md). |
| DiffAN | `diffan/main.py` | Base method DOTS builds on (Sanchez et al., 2023), adapted from the [official implementation](https://github.com/vios-s/DiffAN). |
| SCORE, CAM, DAS, NoGAM | `dodisc/dodiscover-main/dodisc.py` | Via [dodiscover](https://github.com/py-why/dodiscover); `param` selects the method. |
| PCMCI, PCMCI+ | `tigramite/main.py` | Via [tigramite](https://github.com/jakobrunge/tigramite); `param` selects the variant. |
| VARLiNGAM | `lingam/main.py` | Via the `lingam` package. |
| DYNOTEARS | `dynotears/main.py` | Via `causalnex`. |
| TCDF | `tcdf/TCDF-master/main.py` | Nauta et al., 2019. |
| TiMINo | `timino/main.R` | R algorithm (Peters et al., 2013); configured under `r-algorithms`. |
| dummy | `dummy/dummy.py` | Fully-connected baseline. |

Static methods can additionally be post-processed with **temporal constraints** (no future→past edges, optionally no instantaneous edges) — the `*-const` variants in the configs.

**Data**: a synthetic time-series generator, plus three real-world benchmark suites shipped in `data/` (FinanceCPT, NetSim fMRI, CausalTime) — see [THIRD_PARTY.md](THIRD_PARTY.md) for provenance.

**Metrics**: precision/recall/F1 (and more) computed on the **summary graph** (`s_*`) and, where supported, the **window graph** (`w_*`).

## Repository layout

```
config/               experiment configs (one per experiment; paper_* replicate the paper)
data/                 benchmark datasets (see THIRD_PARTY.md)
analysis/             notebooks producing the paper's figures from workflow results
workflow/
  Snakefile           workflow entry point
  run.sh              example launcher
  rules/              Snakemake rules (data prep, per-algorithm predict/evaluate, plots)
  scripts/            data generation/preparation, evaluation, per-algorithm wrappers
    algorithms/<alg>/ one directory per algorithm (vendored third-party code included)
  envs/               conda environment specifications
```

## Installation

Requires [conda](https://docs.anaconda.com/miniconda/) (Linux recommended; the paper experiments ran on Linux).

**1. Snakemake environment.** The workflow itself runs in an env named `snakemake`, which also needs `pandas` and `scikit-learn` (used when parsing configs):

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake pandas scikit-learn
```

**2. Algorithm environments (created manually, referenced by name in the configs):**

```bash
# dodiscover — ALWAYS required: also used for evaluation and data preparation
conda env create -f workflow/envs/dodiscover.yml
conda run -n dodiscover pip install -e workflow/scripts/algorithms/dodisc/dodiscover-main

# tigramite (PCMCI/PCMCI+) — installed from the vendored source
conda env create -f workflow/envs/tigramite.yml
conda run -n tigramite pip install ./workflow/scripts/algorithms/tigramite/tigramite-master
```

Environments for the remaining algorithms (`DOTS/DiffAN`, `TCDF`, `TiMINo`, …) are referenced **by path** in the configs and created automatically by Snakemake on first use (`--sdm conda`).

> For bit-for-bit replication of the paper's DiffAN/DOTS results, the exact (linux-64) environment export is kept in `workflow/envs/diffan-paper-linux64.yml`.

## Quick start

Run the small smoke-test experiment (synthetic data, fast algorithms only):

```bash
conda activate snakemake
cd workflow
snakemake --cores 2 --sdm conda --configfile ../config/config_test.yml
```

Outputs land in `workflow/results/<results.name>/`:

- `data/…` — generated/prepared dataset instances (`samples.csv`, `samples_lagged.csv`, `graphs_true.json`)
- `predictions/<alg>/…/graphs_hat.json` — estimated graphs (+ `info.csv` with runtime)
- `eval/<alg>/results.csv` — per-algorithm metrics
- `summary/results.csv` — everything combined; this is what the analysis notebooks consume
- `plots/…` — optional box/bar plots per metric (if `results.plots: True`)

A dry run (`-n`) shows the job graph without executing anything.

## Replicating the paper experiments

Each experiment is one config file; run it exactly like the quick start, swapping the `--configfile`. Figures are produced by the corresponding notebook in `analysis/` after the workflow finishes.

| Config | Experiment | Analysis notebook(s) |
|---|---|---|
| `paper_sim.yml` | Synthetic-data comparison (sample size × dimensionality × lags × 10 seeds) | `sim_plots.ipynb` |
| `paper_causaltime.yml` | CausalTime (medical, pm25, traffic) | `causaltime_plots.ipynb` |
| `paper_ablation.yml` | DOTS ablation: number of orderings (`n_ord`) | `dots_ablation.ipynb` |
| `paper_ablation_softvote.yml` | DOTS ablation: soft-voting threshold (`theta`) | `dots_ablation_softvote.ipynb` |
| `paper_ablation_lags.yml` | DOTS ablation: assumed lag vs. true lag | `dots_ablation_lags.ipynb` |
| `paper_generate_data.yml` | (Helper) regenerates the synthetic data behind `data/ablation_lags/` | — |

Notes:

- **Compute**: the full `paper_sim.yml` grid is large (hundreds of runs, deep-learning methods included). A GPU speeds up DiffAN/DOTS/TCDF considerably (enable `cudatoolkit` in `workflow/envs/diffan.yml`). Increase `--cores` to parallelise.
- The orderings-diversity ablation (`analysis/dots_ablation_orders.ipynb`) is not part of the Snakemake workflow; it is driven by the standalone script `workflow/scripts/algorithms/dots/ablation_orders.py`.

## Using the framework for your own benchmarks

### New experiment

Copy a config from `config/` and edit:

- `data.params` — either synthetic-generation parameters (`n`, `features`, `n_lag`, `seed`, …) or a `dataset` block with a preparation script (`data.options.scriptname`) and path.
- `results` — experiment name, metrics (`s_f1`, `w_f1`, …), number of repeats, plots on/off.
- `algorithms` / `r-algorithms` — the roster to run. Every hyperparameter list is expanded into a grid, so `alpha: [0.01, 0.05]` runs both.

### New algorithm

1. Create `workflow/scripts/algorithms/<name>/main.py`. Snakemake injects a `snakemake` object; your script should read `snakemake.input['data']` (the time series, one column per variable) and/or `snakemake.input['data_lag']` (time-lagged design matrix), and write:
   - `snakemake.output['pred']` — the estimated graph, via `save_result_adjmat` from `workflow/scripts/graph_utils.py`;
   - `snakemake.output['info']` — a one-row CSV with a `runtime` column.
2. Add a conda env spec under `workflow/envs/`.
3. Add an entry to your config (`id`, optional `hparams`, `script.file`, `script.env`, optional `script.param` for wrappers that dispatch between variants).

R algorithms follow the same pattern under `r-algorithms` (see `timino/`); their raw output is converted by `workflow/scripts/process_r_output.py`.

### New dataset

Add a preparation script under `workflow/scripts/data/` that writes `samples.csv`, `samples_lagged.csv` and `graphs_true.json` (see `prepare_dataset.py` for the expected format), then reference it from a config's `data.options.scriptname`.

## License and third-party components

The benchmark's own code is released under the [Apache 2.0 license](LICENSE). The repository vendors several third-party algorithm implementations and datasets that remain under their own licenses — see [THIRD_PARTY.md](THIRD_PARTY.md) before reusing those components.

## Projects that used TSCD Benchmark
1. Causal Ordering for Structure Learning from Time Series [[paper](https://openreview.net/forum?id=hWuTzqggSd)][[code](https://github.com/CHAI-UK/DOTS)]
2. Rethinking Chronological Causal Discovery with Signal Processing [[paper](https://doi.org/10.1109/IEEECONF67917.2025.11443432)]