# Exporta conjuntos de dados de pacotes do CRAN (tarball-fonte, sem instalar o pacote) ou do R base, como CSV.
# Passo 45 do plano de revisão. Valores exportados como estão; NA continua NA.
#
# Uso:
#   Rscript exportar_r.R cran <pacote> <objeto> <dir_saida>   # baixa o tarball atual do CRAN, extrai data/ e exporta
#   Rscript exportar_r.R base <objeto> <dir_saida>            # conjunto do pacote datasets do R base
args <- commandArgs(trailingOnly = TRUE)
repo <- "https://cloud.r-project.org"
mode <- args[1]
if (mode == "cran") {
  pkg <- args[2]; obj <- args[3]; out <- args[4]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  ap <- available.packages(repos = repo)
  version <- ap[pkg, "Version"]
  tarball <- file.path(out, sprintf("%s_%s.tar.gz", pkg, version))
  if (!file.exists(tarball)) download.file(sprintf("%s/src/contrib/%s_%s.tar.gz", repo, pkg, version), tarball, mode = "wb")
  untar(tarball, exdir = out)
  env <- new.env()
  for (f in list.files(file.path(out, pkg, "data"), pattern = "\\.(rda|RData|rdata)$", full.names = TRUE)) load(f, envir = env)
  if (!exists(obj, envir = env)) stop(sprintf("objeto %s não encontrado em %s", obj, pkg))
  write.csv(as.data.frame(get(obj, envir = env)), file.path(out, sprintf("%s.csv", obj)), row.names = FALSE, na = "")
  cat(sprintf("%s\t%s\t%s\n", pkg, version, basename(tarball)))
} else if (mode == "base") {
  obj <- args[2]; out <- args[3]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  write.csv(get(obj, envir = asNamespace("datasets")), file.path(out, sprintf("%s.csv", obj)), row.names = FALSE, na = "")
  cat(sprintf("datasets\t%s\tR base\n", R.version.string))
} else {
  stop("modo desconhecido")
}
