import numpy as np

# create a block diagonal matrix with block size 3 of the same size as true_causal_matrix
def get_instantaneous_effect_matrix_constraint(nb_timesteps, nb_variables):
    return 1- np.kron(np.eye(nb_timesteps), np.ones((nb_variables, nb_variables)))

def get_temporal_effect_matrix_constraint(nb_timesteps, nb_variables):
    return 1 - np.triu(np.ones((nb_variables*nb_timesteps, nb_variables*nb_timesteps)), k=1)

def apply_temporal_constraint(adj, nb_timesteps, nb_variables, instantanous_constraint: bool = True):
    constraint_matrix_temp = get_temporal_effect_matrix_constraint(nb_timesteps=nb_timesteps, nb_variables=nb_variables)
    # get intersectio between constraint_matrix_inst and constraint_matrix_temp
    if instantanous_constraint:
        constraint_matrix_inst = get_instantaneous_effect_matrix_constraint(nb_timesteps=nb_timesteps, nb_variables=nb_variables)
        constraint_all = np.logical_and(constraint_matrix_inst, constraint_matrix_temp)
    else:
        constraint_all = constraint_matrix_temp

    adj_constrained = np.logical_and(adj, constraint_all).astype(int)
    return adj_constrained