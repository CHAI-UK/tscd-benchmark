import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(snakemake.input[0])

title = ''
first = True
for k_param in snakemake.params['dparams']:
    df = df.loc[df[k_param] == snakemake.params['dparams'][k_param]]
    if first:
        title = f'{k_param}={snakemake.params['dparams'][k_param]}'
        first = False
    else:
        title += f' | {k_param}={snakemake.params['dparams'][k_param]}'

sns.boxplot(data=df, x=snakemake.params['dmetric'], y='id', hue='id').set(ylabel='', title=title)
plt.savefig(snakemake.output['box'])
plt.close()

sns.barplot(data=df, x=snakemake.params['dmetric'], y='id', hue='id').set(ylabel='', title=title)
plt.savefig(snakemake.output['bar'])
plt.close()