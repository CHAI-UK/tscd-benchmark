import os
import sys

import numpy as np
import pandas as pd

sys.path.append('..')
from graph_utils import string_nodes, three_col_format_to_graphs, save_result_tgraph


def rename_lagged_cols(col):
    sp = col.split('_')
    return f"{sp[0]}_lag{sp[1]}"
    

data_fname = os.path.join(snakemake.config['data']['options']['path'], f"data/data_{snakemake.params['opt']['seed']}.csv")
data = pd.read_csv(data_fname, index_col=False, header=None)
nodes = string_nodes(data.columns)

graph_fname = os.path.join(snakemake.config['data']['options']['path'], f"graphs/graph_{snakemake.params['opt']['seed']}.csv")
ground_truth = np.loadtxt(graph_fname, delimiter=',')

tgtrue, max_lag = three_col_format_to_graphs(nodes, ground_truth)
save_result_tgraph(tgtrue, snakemake.output['graphs'])

data.columns = nodes

# create lagged samples
lag_list = list(range(0, max_lag+1))
data_lagged = data.shift(periods=lag_list)[max_lag:]

data_lagged.columns = data_lagged.columns.map(rename_lagged_cols)

data.to_csv(snakemake.output['data'], index=False)
data_lagged.to_csv(snakemake.output['data_lagged'], index=False)