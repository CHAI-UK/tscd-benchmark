conda create -n snakemake -c conda-forge -c bioconda snakemake pandas scikit-learn

conda activate snakemake

cd workflow

snakemake --cores 2 --sdm conda --configfile ../config/config_test.yml