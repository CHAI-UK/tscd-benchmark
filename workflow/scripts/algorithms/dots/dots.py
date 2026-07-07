import sys
from pathlib import Path

import torch
import numpy as np

from collections import Counter
from itertools import permutations, product
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# DOTS builds on the DiffAN implementation in the sibling `diffan` directory
# (score model training, topological ordering and CAM pruning are reused).
_pkg_dir = Path(__file__).resolve().parent
_diffan_dir = _pkg_dir.parent / 'diffan'
_scripts_dir = _pkg_dir.parents[1]
for _p in (str(_diffan_dir), str(_scripts_dir)):
    if _p not in sys.path:
        sys.path.append(_p)

from diffan import DiffAN, apply_temporal_constraint
from algorithms.diffan.utils import full_DAG


def get_transitive_closure_from_orderings(topological_orderings):
    full_DAGs = [full_DAG(ordering) for ordering in topological_orderings]
    transitive_closure = np.logical_and.reduce(full_DAGs).astype(int)
    return transitive_closure, full_DAGs

def get_soft_transitive_closure_from_orderings(topological_orderings):
    full_DAGs = [full_DAG(ordering) for ordering in topological_orderings]
    transitive_closure_soft = np.sum(full_DAGs, axis=0).astype(int) > 0
    return transitive_closure_soft

def get_soft_transitive_closure_from_orderings_01(topological_orderings, theta=0.0):
    full_DAGs = [full_DAG(ordering) for ordering in topological_orderings]
    transitive_closure_soft = np.mean(full_DAGs, axis=0) > theta
    return transitive_closure_soft

# The function all_possible_permutations generates all possible permutations of the flattened list
# where elements are only permuted within their respective inner lists.
def all_possible_permutations(list_of_lists):
    # Generate all permutations for each inner list
    perms = [list(permutations(inner_list)) for inner_list in list_of_lists]
    # Compute the Cartesian product of these permutations
    product_of_perms = product(*perms)
    # Flatten each combination of permutations and collect the results
    result = [ [item for perm in perm_tuple for item in perm] for perm_tuple in product_of_perms ]
    return result


class DOTS(DiffAN):
    """DOTS: diffusion-based causal discovery for time series.

    Extends DiffAN (Sanchez et al., 2023) to lagged time series data by
    computing multiple topological orderings across diffusion steps,
    aggregating them via soft voting (threshold `theta`), and enforcing
    temporal constraints on the resulting window graph.
    """

    def __init__(self, n_nodes, masking = True, residue= True,
                epochs: int = int(3e3), batch_size : int = 1024, learning_rate : float = 0.001, nn_depth_mul=3, steps=1e2, esw=300):
        super().__init__(n_nodes, masking, residue, epochs, batch_size, learning_rate, nn_depth_mul, steps, esw)

    def fit(self, X, constraint=True, inst_const=True, nb_timesteps=-1, nb_variables=-1, nb_orderings=10, norm=True, theta=0.0):
        if norm:
            X = self.normalize_data(X)
        else:
            X = torch.FloatTensor(X).to(self.device)
        # training (0.8); sample
        self.train_score(X)
        # inference (0.2)
        steps_list = list(range(0, self.n_steps+1, self.n_steps//nb_orderings))
        orders = list()
        for step in steps_list:
            order = self.topological_ordering(X, step=step)
            orders.append(order)
        adj_matrix = get_soft_transitive_closure_from_orderings_01(orders, theta) # theta - soft-voting
        adj = self.pruning_adj(adj_matrix.astype(int), X.detach().cpu().numpy())
        if constraint:
            adj = apply_temporal_constraint(adj, nb_timesteps, nb_variables, inst_const)
        return adj

    def fit_all_orders(self, X, inst_const=True, nb_timesteps=-1, nb_variables=-1, nb_orderings=10):
        X = self.normalize_data(X)
        self.train_score(X)
        steps_list = list(range(0, self.n_steps+1, self.n_steps//nb_orderings))
        orders = list()
        results = []
        for step in steps_list:
            order = self.topological_ordering(X, step=step)
            orders.append(order)
            dag = full_DAG(order)
            adj = self.pruning_adj(dag.astype(int), X.detach().cpu().numpy())
            adj = apply_temporal_constraint(adj, nb_timesteps, nb_variables, inst_const)
            results.append(adj)

        adj_matrix = get_soft_transitive_closure_from_orderings(orders)
        adj = self.pruning_adj(adj_matrix.astype(int), X.detach().cpu().numpy())
        adj = apply_temporal_constraint(adj, nb_timesteps, nb_variables, inst_const)
        results.append(adj)

        return results

    def get_orders(self, X, nb_orderings):
        X = self.normalize_data(X)
        self.train_score(X)
        steps_list = list(range(0, self.n_steps+1, self.n_steps//nb_orderings))
        orders = list()
        for step in steps_list:
            order = self.topological_ordering(X, step=step)
            orders.append(order)

        return orders

    def fit_temporal_gmm(self, X, nb_timesteps, nb_variables):
        X = self.normalize_data(X)
        self.train_score(X)
        order_chuncks = self.multiple_topological_orderings_gmm(X)
        all_orderings = all_possible_permutations(order_chuncks)
        adj_matrix = get_soft_transitive_closure_from_orderings(all_orderings)
        adj_matrix_pruned = self.pruning_adj(adj_matrix, X.detach().cpu().numpy())
        adj_matrix_pruned_temporal = apply_temporal_constraint(adj_matrix_pruned, nb_timesteps, nb_variables)
        return adj_matrix_pruned_temporal

    def multiple_topological_orderings_gmm(self, X, eval_batch_size = None):

        if eval_batch_size is None:
            eval_batch_size = self.batch_size
        eval_batch_size = min(eval_batch_size, X.shape[0])

        X = X[:self.batch_size]
        steps_list = range(0, self.n_steps+1, self.n_steps//self.n_votes)

        self.model.eval()
        order = list()
        active_nodes = list(range(self.n_nodes))

        with tqdm(total=len(active_nodes) - 1, desc="Ordering Nodes") as pbar:
            while len(active_nodes) > 1:
                leaves, inner_order = self.compute_hessian_get_leaves(X, steps_list, eval_batch_size, order, active_nodes)
                order.append(inner_order)
                active_nodes = [n for idx, n in enumerate(active_nodes) if idx not in leaves]
                print(f'Inner order: {inner_order}')
                pbar.update(1)
        order.append(active_nodes)
        order.reverse()
        return order

    def compute_hessian_get_leaves(self, X, steps_list, eval_batch_size, order, active_nodes):
        leaves_per_step = list()
        for step in steps_list:
            model_fn_functorch = self.get_model_function_with_residue(step, active_nodes, order)
            data_loader = torch.utils.data.DataLoader(X, eval_batch_size, drop_last=True, shuffle=False)
            hessian_diag_var = self.compute_hessian_diagonal(data_loader, active_nodes, model_fn_functorch)
            leaves = self.get_leaves(hessian_diag_var)
            leaves_per_step.append(leaves)
        # get intersection of leaves in leaves_per_step which is a list of lists
        all_leaves = [item for sublist in leaves_per_step for item in sublist]
        leaves_count = Counter(all_leaves)
        sorted_leaves = sorted(leaves_count, key=leaves_count.get, reverse=True)
        unique_leaves = sorted_leaves[:2] if len(sorted_leaves) > 2 else sorted_leaves
        print(f'Unique leaves: {unique_leaves}')
        inner_order = [active_nodes[n] for n in unique_leaves]
        return unique_leaves, inner_order

    def get_leaves(self, hess_var):
        data = hess_var.reshape(-1, 1)  # Reshape data for sklearn (if necessary)
        # Range of components to try
        n_components_range = range(2, max(data.shape[0]//2,3))

        lowest_bic = np.inf
        best_gmm = None

        # Fit GMMs with different numbers of components and select the best one
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=0)
            gmm.fit(data)
            bic = gmm.bic(data)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm

        # Extract the means of the components
        component_means = best_gmm.means_.flatten()

        # Identify the component with the lowest mean
        min_mean_component = np.argmin(component_means)

        # Predict component assignments for each data point
        component_assignments = best_gmm.predict(data)

        # Select data points assigned to the component with the lowest mean
        selected_indices = np.where(component_assignments == min_mean_component)[0]
        # select the 2 smallest data components with selected_indices
        selected_indices = np.argsort(data[selected_indices].flatten())[:2]
        return selected_indices
