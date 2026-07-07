import pandas as pd

df_all = None
for path in snakemake.input["results"]:
    df = pd.read_csv(path)
    df_all = pd.concat([df_all, df], axis=0)

df_all.to_csv(snakemake.output[0], index=False)