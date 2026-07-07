import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

sys.path.append('..')
from graph_utils import save_result_graph

def process(arr, s_pad=1):
    n_pad = arr.shape[0] - 1
    new_arr = np.zeros(((arr.shape[0]*arr.shape[1])+(n_pad*s_pad), arr.shape[2]))

    row_step = arr.shape[1]
    for i in range(arr.shape[0]):
        new_arr[(row_step*i)+(i*s_pad):(row_step*(i+1))+(i*s_pad), :] = arr[i, :, :]
    
    return new_arr

def rename_lagged_cols(col):
    sp = col.split('_')
    return f"{sp[0]}_lag{sp[1]}"

data_fname = os.path.join(snakemake.params['data_opt']['path'], snakemake.params['opt']['seed'], 'gen_data.npy')
data = np.load(data_fname)

graph_fname = os.path.join(snakemake.params['data_opt']['path'], snakemake.params['opt']['seed'], 'graph.npy')
graph = np.load(graph_fname)

# 3D to 2D
data_2d = process(data)

# Drop residual variables
data_2d = data_2d[:, :data_2d.shape[1]//2]

# Create column names
col_names = [f'X{i+1}' for i in range(data_2d.shape[1])]

df_data = pd.DataFrame(data_2d, columns=col_names)

# Create lagged samples
n_lag = snakemake.params['opt']['lag']
lag_list = list(range(0, n_lag+1))
df_data_lagged = df_data.shift(periods=lag_list)[n_lag:]

df_data_lagged.columns = df_data_lagged.columns.map(rename_lagged_cols)

# Save processed data
df_data.to_csv(snakemake.output['data'], index=False)
df_data_lagged.to_csv(snakemake.output['data_lagged'], index=False)

# Graph representation (array to df to DiGraph)
df_graph = pd.DataFrame(graph, columns=col_names, index=col_names)
gtrue = nx.from_pandas_adjacency(df_graph, create_using=nx.DiGraph)

save_result_graph(gtrue, snakemake.output['graphs'])