# DiffAN

This directory contains the implementation of **DiffAN** used as a baseline in the benchmark, adapted from the official implementation by the original authors:

> Pedro Sanchez, Xiao Liu, Alison Q. O'Neil, Sotirios A. Tsaftaris.
> *Diffusion Models for Causal Discovery via Topological Ordering* ICLR 2023.
> Code: https://github.com/vios-s/DiffAN

Adaptations made for this benchmark: a Snakemake entry point (`main.py`), optional temporal constraints on the estimated graph (`apply_temporal_constraint` in `diffan.py`, which can also be applied to the output of any other static algorithm), and minor robustness fixes.

Our own method **DOTS** builds on this implementation but lives in the sibling [`dots/`](../dots) directory — they are evaluated as separate methods in the benchmark.

## Files

- `diffan.py` — the `DiffAN` model and temporal-constraint helpers.
- `main.py` — Snakemake entry point (referenced from the benchmark configs as `diffan/main.py`).
- `gaussian_diffusion.py`, `nn.py` — diffusion process and score network.
- `pruning.py` + `pruning_R_files/` — CAM pruning (runs R via `cdt`; requires the `mgcv` R package).
- `utils.py` — shared helpers.

## Environment

Created from `workflow/envs/diffan.yml` (conda env named `diffan`). This environment is shared with DOTS.
