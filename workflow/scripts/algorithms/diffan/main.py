import time
import pandas as pd

from diffan import DiffAN
from graph_utils import save_result_adjmat

df = pd.read_csv(snakemake.input['data'])
df_lag = pd.read_csv(snakemake.input['data_lag'])

nb_vars = df.shape[1]
nb_nodes = df_lag.shape[1]
nb_lags = nb_nodes // nb_vars

opt = snakemake.params['alg_opt']

model = DiffAN(nb_nodes, masking=True, residue=False,
               learning_rate=opt.get('lr', 0.001),
               batch_size=opt.get('batch_size', 1024),
               nn_depth_mul=opt.get('nn_depth', 3),
               steps=opt.get('steps', 100),
               esw=opt.get('early_stop', 300))

t_start = time.time()

adj = model.fit(df_lag.to_numpy(),
                constraint=opt.get('constraint', False),
                inst_const=opt.get('const_inst', True),
                nb_timesteps=nb_lags,
                nb_variables=nb_vars,
                norm=opt.get('norm', True))

t_end = time.time()
t_delta = t_end - t_start

# lagged adjmat
df_adj = pd.DataFrame(adj, columns=df_lag.columns, index=df_lag.columns)
save_result_adjmat(df_adj, df.columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)
