import sys

from causalnex.structure.transformers import DynamicDataTransformer
from dgp import format_adjmat, generate

sys.path.append('..')
from graph_utils import save_result_adjmat

opt = snakemake.params['opt']
n_lag = int(opt['n_lag'])

graph, data = generate(
    num_features=int(opt['features']),
    num_samples=int(opt['n']),
    min_lag=n_lag,
    max_lag=n_lag,
    seed=int(opt['seed']),
    prob_ar=float(opt.get('prob_ar', 0.3)),
    prob_edge=float(opt.get('prob_edge', 0.4)),
)

data_lagged = DynamicDataTransformer(p=n_lag).fit_transform(data, return_df=True)
adjmat = format_adjmat(graph, data_lagged.columns)

adjmat.to_csv(snakemake.output['adjmat'], index=False)
data.to_csv(snakemake.output['data'], index=False)
data_lagged.to_csv(snakemake.output['data_lagged'], index=False)
save_result_adjmat(adjmat, data.columns, snakemake.output['graphs'])
