# Evaluation and utility functions for graphs from:
# https://github.com/ckassaad/causal_discovery_for_time_series

import json
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict

def string_nodes(nodes):
    new_nodes = []
    for col in nodes:
        try:
            int(col)
            new_nodes.append("X" + str(col))
        except ValueError:
            new_nodes.append(col)
    return new_nodes

def three_col_format_to_graphs(nodes, three_col_format):
    max_lag = 0
    tgtrue = nx.DiGraph()
    tgtrue.add_nodes_from(nodes)
    for i in range(three_col_format.shape[0]):
        c = "X"+str(int(three_col_format[i, 0]))
        e = "X"+str(int(three_col_format[i, 1]))
        tgtrue.add_edges_from([(c, e)])
        n_lag = int(three_col_format[i, 2])
        tgtrue.edges[c, e]['time'] = [n_lag]

        if n_lag > max_lag:
            max_lag = n_lag

    return tgtrue, max_lag

# Temporal graph to summary graph
def tgraph_to_graph(tg):
    g = nx.DiGraph()
    g.add_nodes_from(tg.nodes)
    for cause, effects in tg.adj.items():
        for effect, _ in effects.items():    
            g.add_edges_from([(cause, effect)])
    return g

# Temporal graph to a list representation
# (for evaluation purposes)
def tgraph_to_list(tg):
    list_tg = []
    for cause, effects in tg.adj.items():
        for effect, eattr in effects.items():
            t_list = eattr['time']
            for t in t_list:
                list_tg.append((cause, effect, t))
    return list_tg

# Adjmat to DiGraph (summary graph)
def dataframe_to_graph(df):
    ghat = nx.DiGraph()
    ghat.add_nodes_from(df.columns)
    for name_x in df.columns:
        if df[name_x].loc[name_x] > 0:
            ghat.add_edges_from([(name_x, name_x)])
        for name_y in df.columns:
            if name_x != name_y:
                if df[name_y].loc[name_x] == 2:
                    ghat.add_edges_from([(name_x, name_y)])
    return ghat

# Dict to DiGraph (temporal window graph)
def dict_to_tgraph(temporal_dict, nodes):
    tghat = nx.DiGraph()
    tghat.add_nodes_from(nodes)
    for name_y in temporal_dict.keys():
        for name_x, t_xy in temporal_dict[name_y]:
            if (name_x, name_y) in tghat.edges:
                tghat.edges[name_x, name_y]['time'].append(-t_xy)
            else:
                tghat.add_edges_from([(name_x, name_y)])
                tghat.edges[name_x, name_y]['time'] = [-t_xy]
    return tghat

def process_parents(parents):
    pa_t = defaultdict(list)
    for p in parents:
        # X1_lag0
        p_split = p.split('_')
        name = p_split[0]
        # 'lag' is always 3 chars, followed by a number
        lag = int(p_split[1][3:])
        pa_t[name].append(lag)
    return pa_t

def tadjmat_to_tgraph(adjmat, nodes):
    g_conv = nx.DiGraph()
    # use non-lagged names
    g_conv.add_nodes_from(nodes)
    n_features = len(nodes)

    # check only parents of non-lagged nodes
    for child in adjmat.columns[:n_features]:
        child_name = child.split('_')[0]
        pa = adjmat.loc[adjmat[child] > 0, child].index.to_list()
        pa_d = process_parents(pa)
        for pa_key in pa_d:
            g_conv.add_edges_from([(pa_key, child_name)])
            g_conv.edges[pa_key, child_name]['time'] = pa_d[pa_key]
    return g_conv

def graph_to_dict(g):
    return json_graph.adjacency_data(g)

def dict_to_graph(d):
    return json_graph.adjacency_graph(d)

def save_json(data, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file)
    
def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_result_adjmat(adjmat, nodes, filepath):
    # convert adjmat to (temporal) DiGraph (window graph)
    tg = tadjmat_to_tgraph(adjmat, nodes)
    save_result_tgraph(tg, filepath)

def save_result_tgraph(tghat, filepath):
    # summary graph
    ghat = tgraph_to_graph(tghat)
    tg_d = graph_to_dict(tghat)
    g_d = graph_to_dict(ghat)

    d = {}
    d['tgraph'] = tg_d
    d['graph'] = g_d

    # both graphs saved in a single json file
    save_json(d, filepath)

def save_result_graph(ghat, filepath):
    g_d = graph_to_dict(ghat)

    d = {}
    d['graph'] = g_d

    # both graphs saved in a single json file
    save_json(d, filepath)

def save_result_temporal(t_dict, nodes, filepath):
    tghat = dict_to_tgraph(t_dict, nodes)
    save_result_tgraph(tghat, filepath)

def save_result_nontemporal(df, filepath):
    ghat = dataframe_to_graph(df)
    save_result_graph(ghat, filepath)

class ModelEvaluation:
    def __init__(self, ghat):
        super(ModelEvaluation, self).__init__()
        self.ghat = ghat

    # [all_oriented, all_adjacent]
    def evaluation(self, gtrue, mode="all_oriented"):
        all_results = {}

        all_results['precision'] = self._precision(gtrue, method=mode)
        all_results['recall'] = self._recall(gtrue, method=mode)
        all_results['f1'] = self._f1(gtrue, method=mode)

        return all_results

    def _hamming_distance(self, gtrue):
        # todo: check if it's correct (maybe it's not truely hamming distance)
        res = nx.graph_edit_distance(self.ghat, gtrue)
        return 1 - res/max(self.ghat.number_of_edges(), gtrue.number_of_edges())

    def _tp(self, gtrue, method="all_oriented"):  # oriented or adjacent
        if method == "all_oriented":
            tp = nx.intersection(gtrue, self.ghat)
        elif method == "all_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.ghat.to_undirected()
            tp = nx.intersection(undirected_true, undirected_hat)
        else:
            raise AttributeError(method)
        return len(tp.edges)

    def _fp(self, gtrue, method="all_oriented"):  # oriented or adjacent
        if method == "all_oriented":
            fp = nx.difference(self.ghat, gtrue)
        elif method == "all_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.ghat.to_undirected()
            fp = nx.difference(undirected_hat, undirected_true)
        else:
            raise AttributeError(method)
        return len(fp.edges)

    def _fn(self, gtrue, method="all_oriented"):  # oriented or adjacent
        if method == "all_oriented":
            fn = nx.difference(gtrue, self.ghat)
        elif method == "all_adjacent":
            undirected_true = gtrue.to_undirected()
            undirected_hat = self.ghat.to_undirected()
            fn = nx.difference(undirected_true, undirected_hat)
        else:
            raise AttributeError(method)
        return len(fn.edges)

    def _topology(self, gtrue, method="all_oriented"):
        correct = self._tp(gtrue, method)
        added = self._fp(gtrue, method)
        missing = self._fn(gtrue, method)
        return correct/(correct + missing + added)

    def _false_positive_rate(self, gtrue, method="all_oriented"):
        true_pos = self._tp(gtrue, method)
        false_pos = self._fp(gtrue, method)
        if false_pos == 0:
            return 0
        else:
            return false_pos / (true_pos + false_pos)

    def _precision(self, gtrue, method="all_oriented"):
        true_pos = self._tp(gtrue, method)
        false_pos = self._fp(gtrue, method)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_pos)

    def _recall(self, gtrue, method="all_oriented"):
        true_pos = self._tp(gtrue, method)
        false_neg = self._fn(gtrue, method)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_neg)

    def _f1(self, gtrue, method="all_oriented"):
        p = self._precision(gtrue, method)
        r = self._recall(gtrue, method)
        if (p == 0) and (r == 0):
            return 0
        else:
            return 2 * p * r / (p + r)

class TemporalModelEvaluation():
    def __init__(self, tghat):
        super(TemporalModelEvaluation, self).__init__()
        self.tghat = tghat

    def evaluation(self, tgtrue):
        all_results = {}

        all_results['precision'] = self._temporal_precision(tgtrue)
        all_results['recall'] = self._temporal_recall(tgtrue)
        all_results['f1'] = self._temporal_f1(tgtrue)

        return all_results

    def _temporal_tp(self, tgtrue):
        list_tg_true = tgraph_to_list(tgtrue)
        list_tg_hat = tgraph_to_list(self.tghat)
        tp = set(list_tg_true).intersection(list_tg_hat)
        return len(tp)

    def _temporal_fp(self, tgtrue):
        list_tg_true = tgraph_to_list(tgtrue)
        list_tg_hat = tgraph_to_list(self.tghat)
        fp = set(list_tg_hat).difference(list_tg_true)
        return len(fp)

    def _temporal_fn(self, tgtrue):
        list_tg_true = tgraph_to_list(tgtrue)
        list_tg_hat = tgraph_to_list(self.tghat)
        fn = set(list_tg_true).difference(list_tg_hat)
        return len(fn)

    def _temporal_false_positive_rate(self, tgtrue):
        true_pos = self._temporal_tp(tgtrue)
        false_pos = self._temporal_fp(tgtrue)
        return false_pos / (true_pos + false_pos)

    def _temporal_precision(self, tgtrue):
        true_pos = self._temporal_tp(tgtrue)
        false_pos = self._temporal_fp(tgtrue)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_pos)

    def _temporal_recall(self, tgtrue):
        true_pos = self._temporal_tp(tgtrue)
        false_neg = self._temporal_fn(tgtrue)
        if true_pos == 0:
            return 0
        else:
            return true_pos / (true_pos + false_neg)

    def _temporal_f1(self, tgtrue):
        p = self._temporal_precision(tgtrue)
        r = self._temporal_recall(tgtrue)
        if (p == 0) and (r == 0):
            return 0
        else:
            return 2 * p * r / (p + r)