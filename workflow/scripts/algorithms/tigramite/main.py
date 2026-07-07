import sys
import time
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn

sys.path.append('../..')
from graph_utils import save_result_temporal

def get_result(pcmci):
    res_dict = dict()
    for effect in pcmci.all_parents.keys():
        res_dict[pcmci.var_names[effect]] = []
        for cause, t in pcmci.all_parents[effect]:
            res_dict[pcmci.var_names[effect]].append((pcmci.var_names[cause], t))
    return res_dict

if snakemake.params['alg_opt']['cond_test'] == 'cmi_knn':
    cond_test = CMIknn()
elif snakemake.params['alg_opt']['cond_test'] == 'par_corr':
    cond_test = ParCorr()

df = pd.read_csv(snakemake.input['data'])
df_lag = pd.read_csv(snakemake.input['data_lag'])

lag_max = int(len(df_lag.columns) / len(df.columns)) - 1

data_tig = pp.DataFrame(df.values, var_names=df.columns)

pcmci = PCMCI(dataframe=data_tig, cond_ind_test=cond_test, verbosity=0)

t_start = time.time()

if snakemake.params['param'] == 'pcmci':
    pcmci.run_pcmci(tau_min=1, tau_max=lag_max, pc_alpha=snakemake.params['alg_opt']['alpha'])
elif snakemake.params['param'] == 'pcmci_plus':
    pcmci.run_pcmciplus(tau_min=1, tau_max=lag_max, pc_alpha=snakemake.params['alg_opt']['alpha'])

t_end = time.time()
t_delta = t_end - t_start

result = get_result(pcmci)

save_result_temporal(result, df.columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)