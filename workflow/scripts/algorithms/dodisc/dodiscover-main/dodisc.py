import sys
import time
import pandas as pd
import networkx as nx

from dodiscover import make_context
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.cam import CAM
from dodiscover.toporder.das import DAS
from dodiscover.toporder.nogam import NoGAM

sys.path.append('../../..')
from graph_utils import save_result_adjmat

from algorithms.utils import apply_temporal_constraint

df = pd.read_csv(snakemake.input['data'])
df_lag = pd.read_csv(snakemake.input['data_lag'])

context = make_context().variables(data=df_lag).build()

if snakemake.params['param'] == 'score':
    alg = SCORE(eta_G=snakemake.params['alg_opt']['eta_g'],
                eta_H=snakemake.params['alg_opt']['eta_h'],
                alpha=snakemake.params['alg_opt']['alpha'])
    
elif snakemake.params['param'] == 'cam':
    alg = CAM(alpha=snakemake.params['alg_opt']['alpha'])

elif snakemake.params['param'] == 'das':
    alg = DAS(eta_G=snakemake.params['alg_opt']['eta_g'],
              eta_H=snakemake.params['alg_opt']['eta_h'],
              alpha=snakemake.params['alg_opt']['alpha'])
    
elif snakemake.params['param'] == 'nogam':
    alg = NoGAM(ridge_alpha=snakemake.params['alg_opt']['ridge_alpha'],
                ridge_gamma=snakemake.params['alg_opt']['ridge_gamma'],
                eta_G=snakemake.params['alg_opt']['eta_g'],
                eta_H=snakemake.params['alg_opt']['eta_h'],
                alpha=snakemake.params['alg_opt']['alpha'])

else:
    raise ValueError(f"Unsupported algorithm choice, provided '{snakemake.params['param']}'.")

t_start = time.time()

alg.learn_graph(df_lag, context)
adj = nx.to_numpy_array(alg.graph_).astype(int)

# add temporal constraint
if snakemake.params['alg_opt']['constraint']:
    nb_vars = df.shape[1]
    nb_nodes = df_lag.shape[1]
    nb_lags = nb_nodes // nb_vars
    adj = apply_temporal_constraint(adj, nb_lags, nb_vars)

t_end = time.time()
t_delta = t_end - t_start

# lagged adjmat
df_adj = pd.DataFrame(adj, columns=df_lag.columns, index=df_lag.columns)

save_result_adjmat(df_adj, df.columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)