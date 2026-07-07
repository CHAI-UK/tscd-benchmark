library(mgcv)

# {SOURCE_DIR} is substituted by pruning.py with the absolute path to
# the pruning_R_files directory, so sourcing works from any working directory.
source(file.path('{SOURCE_DIR}', 'train_gam.R'), chdir=TRUE)
source(file.path('{SOURCE_DIR}', 'selGam.R'), chdir=TRUE)
source(file.path('{SOURCE_DIR}', 'pruning.R'), chdir=TRUE)

dataset <- read.csv(file='{PATH_DATA}', header=FALSE, sep=",")
dag <- read.csv(file='{PATH_DAG}', header=FALSE, sep=",")
set.seed(42)
pruned_dag <- pruning(dataset, dag, pruneMethod = selGam, pruneMethodPars = list(cutOffPVal = {CUTOFF}, numBasisFcts = 10), output={VERBOSE})

write.csv(as.matrix(pruned_dag), row.names = FALSE, file = '{PATH_RESULTS}')
