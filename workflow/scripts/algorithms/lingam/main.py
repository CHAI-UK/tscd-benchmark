import sys
import time
import pandas as pd
import numpy as np

from lingam import VARLiNGAM

sys.path.append('../..')
from graph_utils import save_result_temporal

def get_result(dag, names, split_by_sign=True):
    if split_by_sign:
        direction = np.array(np.where(dag))
        signs = np.zeros_like(dag).astype('int64')
        for i, j in direction.T:
            signs[i][j] = np.sign(am[i][j]).astype('int64')
        dag = signs

    dag = np.abs(dag)
    res_dict = dict()
    for e in range(dag.shape[0]):
        res_dict[names[e]] = []
    for c in range(dag.shape[0]):
        for te in range(dag.shape[1]):
            if dag[c][te] == 1:
                e = te%dag.shape[0]
                t = te//dag.shape[0]
                res_dict[names[e]].append((names[c], -t))
    return res_dict
    

df = pd.read_csv(snakemake.input['data'])
df_lag = pd.read_csv(snakemake.input['data_lag'])

lag_max = int(len(df_lag.columns) / len(df.columns)) - 1

model = VARLiNGAM(lags=lag_max, criterion='bic', prune=True)

t_start = time.time()

model.fit(df)

t_end = time.time()
t_delta = t_end - t_start

m = model._adjacency_matrices
am = np.concatenate([*m], axis=1)

adjmat = np.abs(am) > float(snakemake.params['alg_opt']['alpha'])

result = get_result(adjmat, df.columns)

save_result_temporal(result, df.columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)