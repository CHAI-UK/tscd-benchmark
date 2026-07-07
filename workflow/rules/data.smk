if 'dataset' in config['data']['params']:
    rule prepare_data:
        output:
            data=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples.csv",
            data_lagged=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples_lagged.csv",
            graphs=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/graphs_true.json"
        params:
            opt=data_paramspace.instance,
            data_opt=config['data']['options']
        conda:
            'dodiscover'
        script:
            f"../scripts/data/{config['data']['options']['scriptname']}.py"
else:
    rule generate_data:
        output:
            adjmat=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/adjmat.csv",
            data=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples.csv",
            data_lagged=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples_lagged.csv",
            graphs=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/graphs_true.json"
        params:
            opt=data_paramspace.instance
        conda:
            "../envs/causalnex.yml"
        script:
            "../scripts/data/create_synthetic_data.py"