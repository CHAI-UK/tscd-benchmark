import sys
import time
import pandas as pd
import numpy as np

sys.path.append('../..')
from graph_utils import save_result_adjmat

df = pd.read_csv(snakemake.input['data'])
df_lag = pd.read_csv(snakemake.input['data_lag'])

cols = df_lag.columns
p = len(cols)

t_start = time.time()

# Fully connected DAG (set lower triangular to 1; diagonal set to 0)
arr = np.tril(np.ones((p, p)), -1)

t_end = time.time()
t_delta = t_end - t_start

adjmat = pd.DataFrame(arr, columns=cols, index=cols)
save_result_adjmat(adjmat, df.columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)