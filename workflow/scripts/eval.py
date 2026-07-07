import pandas as pd
from graph_utils import ModelEvaluation, TemporalModelEvaluation, read_json, dict_to_graph

graphs_true = read_json(snakemake.input['g_true'])
graphs_hat = read_json(snakemake.input['g_hat'])
df_info = pd.read_csv(snakemake.input['info'])

cols = ['id']
vals = [snakemake.params['id']]

for key, value in snakemake.params['data_opt'].items():
    cols.append(key)
    vals.append(value)

for key, value in snakemake.params['alg_opt'].items():
    cols.append(key)
    vals.append(value)

cols.append('repeat')
vals.append(snakemake.params['repeats']['repeats'])

# all methods return the summary graph
me = ModelEvaluation(dict_to_graph(graphs_hat['graph']))
g_results = me.evaluation(dict_to_graph(graphs_true['graph']))

for key, value in g_results.items():
    # 's_' for summary graph
    cols.append(f's_{key}')
    vals.append(value)

# not all methods return the window graph
# not all datasets support window graphs
if ('tgraph' in graphs_hat) and ('tgraph' in graphs_true):
    tme = TemporalModelEvaluation(dict_to_graph(graphs_hat['tgraph']))
    tg_results = tme.evaluation(dict_to_graph(graphs_true['tgraph']))

    for key, value in tg_results.items():
        # 'w_' for window graph
        cols.append(f'w_{key}')
        vals.append(value)

df = pd.DataFrame([vals], columns=cols)
df = pd.concat([df, df_info], axis=1)

df.to_csv(snakemake.output[0], index=False)