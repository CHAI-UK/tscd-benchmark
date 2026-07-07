def make_plots(metric):
    rule:
        name:
            f"plot_{metric}"
        input:
            f"results/{config['results']['name']}/summary/results.csv"
        output:
            box=f"results/{config['results']['name']}/plots/{data_paramspace_noseed.wildcard_pattern}/{metric}_box.pdf",
            bar=f"results/{config['results']['name']}/plots/{data_paramspace_noseed.wildcard_pattern}/{metric}_bar.pdf",
        params:
            dparams=data_paramspace_noseed.instance,
            dmetric=metric
        conda:
            "../envs/plots.yml"
        script:
            "../scripts/plot.py"

for metric in config['results']['metrics']:
    make_plots(metric)