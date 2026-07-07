import pandas as pd
from graph_utils import save_result_nontemporal, save_result_temporal

def dataframe_to_dict(df, names, nlags):
    g_dict = dict()
    for name_y in names:
        g_dict[name_y] = []
    for ty in range(nlags):
        for name_y in names:
            t_name_y = df.columns[ty*len(names)+names.index(name_y)]
            for tx in range(nlags):
                for name_x in names:
                    t_name_x = df.columns[tx * len(names) + names.index(name_x)]
                    if df[t_name_y].loc[t_name_x] == 2:
                        if (name_x, tx-ty) not in g_dict[name_y]:
                            g_dict[name_y].append((name_x, tx - ty))

    return g_dict

temporal=True
if snakemake.params['alg'] == 'timino':
    temporal=False

raw_result = pd.read_csv(snakemake.input['result'])

if temporal:
    df = pd.read_csv(snakemake.input['data'])
    df_lag = pd.read_csv(snakemake.input['data_lag'])
    lag_max = int(len(df_lag.columns) / len(df.columns)) - 1

    tghat = dataframe_to_dict(raw_result, df.columns, lag_max)
    save_result_temporal(tghat, df.columns, snakemake.output[0])
else:
    save_result_nontemporal(raw_result, snakemake.output[0])