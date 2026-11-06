# PKLM (Spohn, Näf, Michel e Meinshausen, 2023) via PKLMtest, em lote.
# Uso: Rscript pklm.R <lista.csv> <saida.csv> <num.proj> <nrep> <semente>
# <lista.csv>: colunas sample_id, path (CSV com a coluna y, que tem NA, e as auxiliares).
# Grava uma linha por instância: sample_id, pklm_p, seconds, status.
suppressPackageStartupMessages(library(PKLMtest))
args <- commandArgs(trailingOnly = TRUE)
items <- read.csv(args[1], stringsAsFactors = FALSE)
out_path <- args[2]
num_proj <- as.integer(args[3])
nrep <- as.integer(args[4])
seed <- as.integer(args[5])
rows <- vector("list", nrow(items))
for (i in seq_len(nrow(items))) {
  t0 <- Sys.time()
  X <- as.matrix(read.csv(items$path[i]))
  set.seed(seed)
  res <- tryCatch(
    {
      invisible(capture.output(p <- PKLMtest(X, num.proj = num_proj, nrep = nrep)))
      list(p = unname(p[1]), status = "ok")
    },
    error = function(e) list(p = NA_real_, status = paste0("failed: ", conditionMessage(e)))
  )
  rows[[i]] <- data.frame(
    sample_id = items$sample_id[i], pklm_p = res$p,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), status = res$status
  )
}
write.csv(do.call(rbind, rows), out_path, row.names = FALSE)
