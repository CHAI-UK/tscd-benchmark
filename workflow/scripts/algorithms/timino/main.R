library(gam)
library(kernlab)
library(gptk)

source("scripts/algorithms/timino/codeTimino/timino_causality.R")
source("scripts/algorithms/timino/codeTimino/util/hammingDistance.R")
source("scripts/algorithms/timino/codeTimino/util/indtestAll.R")
source("scripts/algorithms/timino/codeTimino/util/indtestHsic.R")
source("scripts/algorithms/timino/codeTimino/util/indtestPcor.R")
source("scripts/algorithms/timino/codeTimino/util/TSindtest.R")
source("scripts/algorithms/timino/codeTimino/util/fitting_ts.R")


data <- read.csv(snakemake@input[["data"]])
data_lag <- read.csv(snakemake@input[["data_lag"]])

alpha <- snakemake@params[["alg_opt"]][["alpha"]]
n_lags <- (ncol(data_lag) / ncol(data)) - 1

t_start <- Sys.time()

result <- timino_dag(data, alpha = alpha, max_lag = n_lags, model = traints_linear, indtest = indtestts_crosscov, output = TRUE)

t_end <- Sys.time()
t_delta <- t_end - t_start

result[is.na(result)] <- 3

for (j1 in 1:nrow(result)){
    for (j2 in 1:nrow(result)){
      if (result[j1,j2] == 1){
        result[j1,j2] <- 2
      }
    }
}

for (j1 in 1:nrow(result)){
    for (j2 in 1:nrow(result)){
      if (result[j1,j2] == 2){
        if (result[j2,j1] == 0){
            result[j2,j1] <- 1
        }
      }
      if (j1 == j2){
          result[j1,j2] <- 1
      }
    }
}

for (j1 in 1:nrow(result)){
    for (j2 in 1:nrow(result)){
      if (result[j1,j2] == 3){
        result[j1,j2] <- 2
        result[j2,j1] <- 2
      }
    }
}

write.table(result, snakemake@output[["pred"]], col.names = colnames(data), row.names = colnames(data), sep = ",")

df_info <- data.frame(runtime = c(t_delta))
write.csv(df_info, snakemake@output[["info"]], row.names = FALSE)