import sys
import time
import TCDF
import pandas as pd

sys.path.append('../../..')
from graph_utils import save_result_temporal

cuda=True
nrepochs=snakemake.params['alg_opt']['epochs']
kernel_size=4
levels=snakemake.params['alg_opt']['h_layers'] + 1
loginterval=500
learningrate=snakemake.params['alg_opt']['lr']
optimizername='Adam'
seed=1111
dilation_c=4
significance=0.8

def runTCDF(datafile):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    df_data = pd.read_csv(datafile)

    allcauses = dict()
    alldelays = dict()
    allreallosses=dict()
    allscores=dict()

    columns = list(df_data)
    for c in columns:
        idx = df_data.columns.get_loc(c)
        causes, causeswithdelay, realloss, scores = TCDF.findcauses(c, cuda=cuda, epochs=nrepochs, 
        kernel_size=kernel_size, layers=levels, log_interval=loginterval, 
        lr=learningrate, optimizername=optimizername,
        seed=seed, dilation_c=dilation_c, significance=significance, file=datafile)

        allscores[idx]=scores
        allcauses[idx]=causes
        alldelays.update(causeswithdelay)
        allreallosses[idx]=realloss

    return allcauses, alldelays, allreallosses, allscores, columns

def tcdf_output_to_adapted_output(allcauses, alldelays, columns):
    g_dict = dict()
    for k in allcauses.keys():
        temp = allcauses[k]
        if temp != []:
            g_dict[columns[k]] = []
            for i in temp:
                g_dict[columns[k]].append((columns[i], -alldelays[(k, i)]))
    return g_dict

t_start = time.time()

allcauses, alldelays, allreallosses, allscores, columns = runTCDF(snakemake.input['data'])

t_end = time.time()
t_delta = t_end - t_start

result = tcdf_output_to_adapted_output(allcauses, alldelays, columns)

save_result_temporal(result, columns, snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)