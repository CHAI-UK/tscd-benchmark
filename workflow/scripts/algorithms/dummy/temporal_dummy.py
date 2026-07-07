import sys
import time
import pandas as pd
import numpy as np

sys.path.append('../..')
from graph_utils import save_result_adjmat

def get_instantaneous_matrix(nb_timesteps, nb_variables):
    return 1- np.kron(np.eye(nb_timesteps), np.ones((nb_variables, nb_variables)))

def get_temporal_matrix(nb_timesteps, nb_variables):
    return 1 - np.triu(np.ones((nb_variables*nb_timesteps, nb_variables*nb_timesteps)), k=1)

df = pd.read_csv(snakemake.input['data'])
df_lag = pd.read_csv(snakemake.input['data_lag'])

nb_vars = df.shape[1]
nb_nodes = df_lag.shape[1]
nb_lags = nb_nodes // nb_vars

cols = df_lag.columns

t_start = time.time()

# Fully connected temporal DAG
arr_inst = get_instantaneous_matrix(nb_lags, nb_vars)
arr_temp = get_temporal_matrix(nb_lags, nb_vars)
arr = np.logical_and(arr_inst, arr_temp).astype(int)

t_end = time.time()
t_delta = t_end - t_start

adjmat = pd.DataFrame(arr, columns=cols, index=cols)
save_result_adjmat(adjmat, df.columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)