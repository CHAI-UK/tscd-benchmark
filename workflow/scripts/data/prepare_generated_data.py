import os
import shutil

import pandas as pd

def rename_lagged_cols(col):
    sp = col.split('_')
    return f"{sp[0]}_lag{sp[1]}"

data_fname = os.path.join(snakemake.params['data_opt']['path'], snakemake.params['opt']['seed'], 'samples.csv')
graph_fname = os.path.join(snakemake.params['data_opt']['path'], snakemake.params['opt']['seed'], 'graphs_true.json')
data = pd.read_csv(data_fname)

max_lag = snakemake.params['opt']['lag']

# create lagged samples
lag_list = list(range(0, max_lag+1))
data_lagged = data.shift(periods=lag_list)[max_lag:]
data_lagged.columns = data_lagged.columns.map(rename_lagged_cols)

shutil.copyfile(data_fname, snakemake.output['data'])
data_lagged.to_csv(snakemake.output['data_lagged'], index=False)
shutil.copyfile(graph_fname, snakemake.output['graphs'])