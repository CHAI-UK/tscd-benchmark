import sys
import time

import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR

sys.path.append('../..')
from graph_utils import save_result_nontemporal

def summary_transform(pred, opt):
    if opt == "max":
        prediction = pred.max(axis=2)
    elif opt == "mean":
        prediction = pred.mean(axis=2)
    return prediction


def var_baseline(d, max_lag):
    """
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    n_vars = d.values.shape[-1]
    #d.index = pd.DatetimeIndex(d.index.values, freq=d.index.inferred_freq)
    
    try:
        # fit var with appropriate max lags
        res = VAR(d).fit(max_lag)
        pred = res.params[1:]        
        pred = np.abs(pred)

        pred = np.stack(
            [pred.values[:, x].reshape(max_lag, n_vars).T for x in range(pred.shape[1])]
        )
    except:
        pred = np.zeros((n_vars, n_vars, max_lag))
        print("Fitting failed")

    out = summary_transform(pred, 'max')
    
    return out


df = pd.read_csv(snakemake.input['data'])
cols = df.columns

t_start = time.time()

arr = var_baseline(df, snakemake.params['alg_opt']['max_lag'])

t_end = time.time()
t_delta = t_end - t_start

adjmat = pd.DataFrame(arr, columns=cols, index=cols)
save_result_nontemporal(adjmat, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)