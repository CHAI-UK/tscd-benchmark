def create_alg_rules(alg):
    if 'hparams' in alg:
        alg_paramspace = Paramspace(yaml2grid(get_alg_hparams(alg['id'])))
        cond_output = f"{alg_paramspace.wildcard_pattern}/{data_paramspace.wildcard_pattern}/{repeats_paramspace.wildcard_pattern}"
        summarise_input = expand(f"results/{config['results']['name']}/eval/{alg['id']}/{{hparams}}/{{dparams}}/{{rparams}}/results.csv", hparams=alg_paramspace.instance_patterns, dparams=data_paramspace.instance_patterns, rparams=repeats_paramspace.instance_patterns)
        use_hps = True
    else:
        alg_paramspace = {}
        cond_output = f"{data_paramspace.wildcard_pattern}/{repeats_paramspace.wildcard_pattern}"
        summarise_input = expand(f"results/{config['results']['name']}/eval/{alg['id']}/{{dparams}}/{{rparams}}/results.csv", dparams=data_paramspace.instance_patterns, rparams=repeats_paramspace.instance_patterns)
        use_hps = False

    rule:
        name:
            f"predict_{alg['id']}"
        input:
            data=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples.csv",
            data_lag=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples_lagged.csv"
        output:
            pred=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/graphs_hat.json",
            info=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/info.csv"
        params:
            alg_opt=lambda w: alg_paramspace.instance if use_hps else {},
            param=lambda w: alg['script']['param'] if 'param' in alg['script'] else ''
        conda:
            lambda w: alg['script']['env'] if 'env' in alg['script'] else 'snakemake'
        script:
            f"../../scripts/algorithms/{alg['script']['file']}"
    
    rule:
        name:
            f"evaluate_{alg['id']}"
        input:
            g_true=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/graphs_true.json",
            g_hat=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/graphs_hat.json",
            info=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/info.csv"
        output:
            f"results/{config['results']['name']}/eval/{alg['id']}/{cond_output}/results.csv"
        params:
            data_opt=data_paramspace.instance,
            alg_opt=lambda w: alg_paramspace.instance if use_hps else {},
            id=f"{alg['id']}",
            repeats=repeats_paramspace.instance
        conda:
            '../../envs/benchmark.yml'
        script:
            "../../scripts/eval.py"
    
    rule:
        name:
            f"summarise_{alg['id']}"
        input:
            results=summarise_input
        output:
            f"results/{config['results']['name']}/eval/{alg['id']}/results.csv"
        script:
            "../../scripts/summarise.py"

def create_r_alg_rules(alg):
    if 'hparams' in alg:
        alg_paramspace = Paramspace(yaml2grid(get_r_alg_hparams(alg['id'])))
        cond_output = f"{alg_paramspace.wildcard_pattern}/{data_paramspace.wildcard_pattern}/{repeats_paramspace.wildcard_pattern}"
        summarise_input = expand(f"results/{config['results']['name']}/eval/{alg['id']}/{{hparams}}/{{dparams}}/{{rparams}}/results.csv", hparams=alg_paramspace.instance_patterns, dparams=data_paramspace.instance_patterns, rparams=repeats_paramspace.instance_patterns)
        use_hps = True
    else:
        alg_paramspace = {}
        cond_output = f"{data_paramspace.wildcard_pattern}/{repeats_paramspace.wildcard_pattern}"
        summarise_input = expand(f"results/{config['results']['name']}/eval/{alg['id']}/{{dparams}}/{{rparams}}/results.csv", dparams=data_paramspace.instance_patterns, rparams=repeats_paramspace.instance_patterns)
        use_hps = False

    rule:
        name:
            f"predict_{alg['id']}"
        input:
            data=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples.csv",
            data_lag=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples_lagged.csv"
        output:
            pred=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/raw_result.csv",
            info=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/info.csv"
        params:
            alg_opt=lambda w: alg_paramspace.instance if use_hps else {},
            param=lambda w: alg['script']['param'] if 'param' in alg['script'] else ''
        conda:
            lambda w: alg['script']['env'] if 'env' in alg['script'] else 'snakemake'
        script:
            f"../../scripts/algorithms/{alg['script']['file']}"
    
    rule:
        name:
            f"process_{alg['id']}"
        input:
            data=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples.csv",
            data_lag=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/samples_lagged.csv",
            result=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/raw_result.csv"
        output:
            f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/graphs_hat.json"
        params:
            alg=alg['id']
        conda:
            '../../envs/benchmark.yml'
        script:
            "../../scripts/process_r_output.py"

    rule:
        name:
            f"evaluate_{alg['id']}"
        input:
            g_true=f"results/{config['results']['name']}/data/{data_paramspace.wildcard_pattern}/graphs_true.json",
            g_hat=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/graphs_hat.json",
            info=f"results/{config['results']['name']}/predictions/{alg['id']}/{cond_output}/info.csv"
        output:
            f"results/{config['results']['name']}/eval/{alg['id']}/{cond_output}/results.csv"
        params:
            data_opt=data_paramspace.instance,
            alg_opt=lambda w: alg_paramspace.instance if use_hps else {},
            id=f"{alg['id']}",
            repeats=repeats_paramspace.instance
        conda:
            '../../envs/benchmark.yml'
        script:
            "../../scripts/eval.py"
    
    rule:
        name:
            f"summarise_{alg['id']}"
        input:
            results=summarise_input
        output:
            f"results/{config['results']['name']}/eval/{alg['id']}/results.csv"
        script:
            "../../scripts/summarise.py"

# Using a function to work around a bug:
# https://github.com/snakemake/snakemake/issues/2178
if 'algorithms' in config:
    for alg in config.get('algorithms'):
        create_alg_rules(alg)

if 'r-algorithms' in config:
    for r_alg in config.get('r-algorithms'):
        create_r_alg_rules(r_alg)