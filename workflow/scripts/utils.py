import pandas as pd
from sklearn.model_selection import ParameterGrid

def yaml2grid(data_orig, exclude=[]):
    data = data_orig.copy()

    for e in exclude:
        data.pop(e, None)

    if 'seed' in data:
        if 'start' in data['seed'] and 'end' in data['seed']:
            seed_list = list(range(data['seed']['start'], data['seed']['end'] + 1))
            data['seed'] = seed_list
        else:
            data['seed'] = data['seed']['id']

    if 'dataset' in data:
        data['dataset'] = [data['dataset']]

    pg = ParameterGrid(data)
    return pd.DataFrame(list(pg))

def yaml2grid_repeats(data):
    l_repeats = list(range(1, data['repeats'] + 1))
    d_repeats = {'repeats': l_repeats}

    pg = ParameterGrid(d_repeats)
    return pd.DataFrame(list(pg))

def get_alg_hparams(alg):
    for c in config.get('algorithms'):
        if c['id'] == alg:
            return c['hparams']
    return None

def get_r_alg_hparams(alg):
    for c in config.get('r-algorithms'):
        if c['id'] == alg:
            return c['hparams']
    return None

def get_algs():
    algs = []
    if 'algorithms' in config:
        for c in config.get('algorithms'):
            algs.append(c['id'])
    
    if 'r-algorithms' in config:
        for c in config.get('r-algorithms'):
            algs.append(c['id'])
    
    return algs