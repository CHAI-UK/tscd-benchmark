import sys
import pandas as pd
import numpy as np
from castle.metrics import MetricsDAG
from dots import DOTS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=3)

df = pd.read_csv('./test_data/samples.csv')
df_lag = pd.read_csv('./test_data/samples_lagged.csv')
adjmat = pd.read_csv('./test_data/adjmat.csv')
adjmat = adjmat.to_numpy()

nb_vars = df.shape[1]
nb_nodes = df_lag.shape[1]
nb_lags = nb_nodes // nb_vars

for seed in range(100):
    print(f"Starting seed = {seed+1}")

    model = DOTS(nb_nodes, masking=True, residue=False, learning_rate=0.001, batch_size=1024, nn_depth_mul=3, steps=100, esw=300)
    results = model.fit_all_orders(df_lag.to_numpy(), True, nb_lags, nb_vars, 10)

    all_results = []
    for i, adj in enumerate(results):
        metrics = MetricsDAG(adj, B_true=adjmat).metrics
        metrics['index'] = i
        metrics['seed'] = seed+1
        all_results.append(metrics)

    new_df = pd.DataFrame(all_results)
    df_all = pd.read_csv('test_data/ablation_orders.csv')
    df_all = pd.concat([df_all, new_df], axis=0)
    df_all.to_csv('test_data/ablation_orders.csv', index=False)