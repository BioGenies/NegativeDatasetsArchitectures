#' @example 
#' library(biogram)
#' example_seq <- "CAD"
#' aaprop_matrix <- aaprop
#' colnames(aaprop_matrix) <- toupper(colnames(aaprop_matrix))
#' CSAMPPRED_features(example_seq, aaprop_matrix) 
CSAMPPRED_features <- function(sequence, aaprop_matrix) {
  
  # {KLEP840101}{Net charge (Klein et al., 1984)}
  f1_avg_charge <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "KLEP840101", x]))) / length(sequence)
  
  # {EISD840101}{Consensus normalized hydrophobicity scale (Eisenberg, 1984)}
  f2_avg_hydrophobicity <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "EISD840101", x]))) / length(sequence)
  
  # {EISD860102}{Atom-based hydrophobic moment (Eisenberg-McLachlan, 1986)}
  f3_avg_hydrophobic_moment <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "EISD860102", x]))) / length(sequence)
  
  # f4_amphipathicity TODO
  # the amphipathicity was calculated as the ratio between
  # hydrophobic and charged residues
  
  # {KOEP990101}{Alpha-helix propensity derived from designed sequences (Koehl-Levitt, 1999)}
  f5_propensity <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "KOEP990101", x]))) / length(sequence)
  
  # {BHAR880101}{Average flexibility indices (Bhaskaran-Ponnuswamy, 1988)}
  f6_flexibility <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "BHAR880101", x]))) / length(sequence)
  
  # {LEVM780101}{Normalized frequency of alpha-helix, with weights (Levitt, 1978)}
  f7_formation_alpha <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "LEVM780101", x]))) / length(sequence)
  
  # {LEVM780102}{Normalized frequency of beta-sheet, with weights (Levitt, 1978)}
  f8_formation_beta <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "LEVM780102", x]))) / length(sequence)
  
  # {LEVM780103}{Normalized frequency of reverse turn, with weights (Levitt, 1978)}
  f9_formation_loop <- sum(unlist(lapply(sequence, function(x) 
    aaprop_matrix[rownames(aaprop_matrix) == "LEVM780103", x]))) / length(sequence)
  
  c(f1_avg_charge,
    f2_avg_hydrophobicity,
    f3_avg_hydrophobic_moment,
    #f4_,
    f5_propensity, 
    f6_flexibility,
    f7_formation_alpha,
    f8_formation_beta,
    f9_formation_loop)
}

create_features <- function(path) {
  fasta_file <- read.fasta(path)
  do.call(rbind, lapply(fasta_file, function(x) CSAMPPRED_features(x, biogram::aaprop)))
}

create_target <- function(path) {
  fasta_file <- read.fasta(path)
  target <- unlist(lapply(fasta_file, function(x) ifelse(grepl("AMP=1", attr(x, "name")) ,1 , 0)))
  names(target) <- NULL
  target <- as.factor(target)
  
  target
}

create_features("example-data/exemplary_train_dataset.fa")
create_target("example-data/exemplary_train_dataset.fa")
