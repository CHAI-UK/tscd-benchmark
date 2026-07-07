"""Synthetic time series generation from a random structural causal model (SCM).

A random lagged DAG over feature variables (X1, X2, ...) is generated first,
every edge is assigned a randomly parameterised nonlinear function, and data
is then sampled by iterating the SCM forward in time. Each feature node has a
dedicated exogenous Gaussian noise parent (Sx1, Sx2, ...).
"""
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

# Functional forms assigned to graph edges, and their sampling probabilities.
FUNCTIONS = ['piecewise_linear', 'trigonometric']
PROB_FUNCTIONS = [0.5, 0.5]

# Edge weights are drawn from one of these ranges (scaled down by the number
# of parents of a node) so that the series do not explode over time.
WEIGHT_RANGES = [(-0.99, -0.7), (0.7, 0.99)]

# The variance of each noise variable is drawn uniformly from this range.
NOISE_VARIANCE_RANGE = (0.01, 0.05)


def generate(num_features, num_samples, min_lag, max_lag, seed, prob_ar, prob_edge):
    """Generate a random time series causal graph and one dataset sampled from it.

    :param num_features: Number of observed variables.
    :param num_samples: Number of time steps to sample.
    :param min_lag: Minimum lag of a parent node (must be >= 1).
    :param max_lag: Maximum lag of a parent node.
    :param seed: Seed for both graph generation and data sampling.
    :param prob_ar: Probability that a variable is autoregressive.
    :param prob_edge: Probability of an edge between two variables.
    :return: Tuple of the full time causal graph (networkx.DiGraph, including
        noise nodes) and the sampled data (pandas.DataFrame, observed only).
    """
    assert min_lag >= 1, 'Instantaneous effects are not supported.'
    rng = np.random.default_rng(seed)
    graph = generate_graph(rng, num_features, min_lag, max_lag, prob_ar, prob_edge)
    equations = generate_equations(rng, graph)
    # The sampling stage restarts the RNG with the same seed.
    data = sample_dataset(np.random.default_rng(seed), graph, equations, num_samples, max_lag)
    return graph, data


def format_adjmat(graph, columns):
    """Adjacency matrix of the ground-truth graph: noise variables dropped,
    nodes renamed from 'X1_t-2' style to 'X1_lag2', and rows/columns ordered
    to match the given lagged-data columns.
    """
    adjmat = nx.to_pandas_adjacency(graph)
    observed = ~adjmat.index.str.contains('Sx')
    adjmat = adjmat.loc[observed, observed]

    def rename(node):
        if node.endswith('_t'):
            return node[:-2] + '_lag0'
        return node.replace('_t-', '_lag')

    adjmat.index = adjmat.index.map(rename)
    adjmat.columns = adjmat.columns.map(rename)
    return adjmat.loc[columns, columns]


def generate_graph(rng, num_features, min_lag, max_lag, prob_ar, prob_edge,
                   max_parents=3, max_children=2, max_lags_per_parent=3):
    """Generate a random DAG over nodes X{i}_t, X{i}_t-1, ..., X{i}_t-{max_lag},
    plus one noise parent Sx{i}_* per feature node. Cross-variable parents are
    only drawn from lags in [min_lag, max_lag].
    """
    lags = range(max_lag + 1)
    features = [f'X{i}' for i in range(1, num_features + 1)]

    graph = nx.DiGraph()
    graph.add_nodes_from(node_name(var, lag) for var in features for lag in lags)
    graph.add_nodes_from(node_name(f'Sx{i}', lag)
                         for i in range(1, num_features + 1) for lag in lags)

    # Autoregressive chains: X_t-k -> X_t-(k-1) at every lag of a variable.
    # The guard matters: when prob_ar is 0 no RNG values may be consumed.
    if prob_ar > 0.0:
        for var in features:
            if rng.random() < prob_ar:
                for lag in range(1, max_lag + 1):
                    graph.add_edge(node_name(var, lag), node_name(var, lag - 1))

    # Draw cross-variable parents for each variable at time t.
    for var in features:
        child = node_name(var, 0)
        candidates = [node for node in graph.nodes
                      if not node.startswith('S') and not node.startswith(var)
                      and node_lag(node) >= min_lag]
        rng.shuffle(candidates)

        # An autoregressive link counts towards max_parents.
        while graph.in_degree(child) < max_parents and candidates:
            parent = candidates.pop(0)
            # rng.random() is drawn even when the children limit blocks the edge.
            if rng.random() < prob_edge and graph.out_degree(parent) < max_children:
                # Repeat the edge at every lag so the graph is time-invariant.
                p, c = parent, child
                graph.add_edge(p, c)
                while node_lag(p) < max_lag:
                    p, c = lag_node(p), lag_node(c)
                    graph.add_edge(p, c)

            parent_var = node_var(parent)
            if sum(node.startswith(parent_var)
                   for node in graph.predecessors(child)) >= max_lags_per_parent:
                candidates = [node for node in candidates if not node.startswith(parent_var)]

    # Every feature node gets a dedicated exogenous noise parent.
    graph.add_edges_from((node_name(f'Sx{i}', lag), node_name(f'X{i}', lag))
                         for i in range(1, num_features + 1) for lag in lags)

    assert nx.is_directed_acyclic_graph(graph), 'Generated graph is not a DAG.'
    return graph


def generate_equations(rng, graph):
    """Assign a random function to every edge of the graph.

    :return: Dict mapping each variable to {parent node at time t: function}.
    """
    variables = sorted({node_var(node) for node in graph.nodes})
    equations = {}
    for var in variables:
        parents = list(graph.predecessors(node_name(var, 0)))
        ranges = WEIGHT_RANGES
        if len(parents) > 1:
            # Scale so the summed weights stay within the initial ranges.
            ranges = [(low / len(parents), high / len(parents)) for low, high in WEIGHT_RANGES]
        equations[var] = {
            parent: (lambda x: x) if parent.startswith('S') else _random_function(rng, ranges)
            for parent in parents
        }
    return equations


def sample_dataset(rng, graph, equations, num_samples, max_lag):
    """Sample the SCM forward in time and return the observed variables."""
    # Warm-up buffer at the start of every series, dropped again below.
    num_samples += 2 * max_lag

    variables = [var for var in equations if not var.startswith('S')]
    noise_variables = [var for var in equations if var.startswith('S')]

    data = {var: np.zeros(num_samples) for var in variables}
    data.update({var: _sample_noise(rng, num_samples) for var in noise_variables})

    # All parents lie at least one lag back (min_lag >= 1), so the variables
    # within a time step can be filled in any fixed order.
    for t in range(max_lag, num_samples):
        for var in variables:
            for parent, func in equations[var].items():
                data[var][t] += func(data[node_var(parent)][t - node_lag(parent)])

    df = pd.DataFrame({var: data[var] for var in variables}).iloc[2 * max_lag:]
    if not np.all(np.isfinite(df)):
        raise RuntimeError('Data generating process produced non-finite data.')
    return df


def node_name(var, lag):
    """'X1' at lag 2 -> 'X1_t-2'; at lag 0 -> 'X1_t'."""
    return f'{var}_t' if lag == 0 else f'{var}_t-{lag}'


def node_var(node):
    return node.split('_')[0]


def node_lag(node):
    return 0 if node.endswith('t') else int(node.split('-')[-1])


def lag_node(node):
    """Shift a node one step back in time: 'X1_t-1' -> 'X1_t-2'."""
    return node_name(node_var(node), node_lag(node) + 1)


def _random_function(rng, weight_ranges):
    """Sample a random scalar function for one edge."""
    function_type = rng.choice(a=FUNCTIONS, p=PROB_FUNCTIONS)
    if function_type == 'piecewise_linear':
        num_knots = rng.choice(10) + 1
        out_weights = _sample_weights(rng, weight_ranges, num_knots)
        in_weights = _sample_weights(rng, weight_ranges, num_knots)
        offsets = rng.uniform(low=-5.0, high=5.0, size=num_knots)
        bias = rng.uniform(low=-0.1, high=0.1)
        return partial(_piecewise_linear, out_weights=out_weights,
                       in_weights=in_weights, offsets=offsets, bias=bias)
    else:  # trigonometric
        amplitude = _sample_weights(rng, weight_ranges)
        frequency = rng.uniform(low=0.5, high=5.0)
        phase = rng.uniform(low=-np.pi, high=np.pi)
        bias = rng.uniform(low=-0.1, high=0.1)
        return partial(_trigonometric, amplitude=amplitude, frequency=frequency,
                       phase=phase, bias=bias)


def _sample_weights(rng, ranges, num_samples=1):
    """Sample weights, independently choosing one of the ranges per sample."""
    if num_samples == 1:
        return rng.uniform(*rng.choice(ranges))
    return np.array([rng.uniform(*rng.choice(ranges)) for _ in range(num_samples)])


def _sample_noise(rng, num_samples):
    """Sample IID Gaussian noise with a random variance."""
    variance = rng.uniform(*NOISE_VARIANCE_RANGE)
    # Only Gaussian noise is used, but can be extended to other distributions.
    rng.choice(a=['gaussian'], p=[1.0])
    return np.sqrt(variance) * rng.standard_normal(num_samples)


def _piecewise_linear(x, out_weights, in_weights, offsets, bias):
    return np.sum(out_weights * np.maximum(0.0, in_weights * (x - offsets))) + bias


def _trigonometric(x, amplitude, frequency, phase, bias):
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase) + bias
