#!/usr/bin/env bash
# Run the benchmark from the workflow/ directory.
# Pick the experiment via --configfile (see ../config/ for available experiments).
#
# Dry run (shows the jobs that would be executed):
#   snakemake --cores 1 --configfile ../config/config_test.yml -n
#
# If a previous run was interrupted, you may need: snakemake --unlock

snakemake --cores 2 --sdm conda --configfile ../config/config_test.yml
