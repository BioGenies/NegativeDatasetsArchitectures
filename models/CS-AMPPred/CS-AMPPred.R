#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(e1071)
library(seqinr)

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
  
  # f4_amphipathicity
  # the amphipathicity was calculated as the ratio between
  # hydrophobic and charged residues
  charged <- c("r", "k", "h", "d", "e")
  hydrophobic <- c("a", "i", "l", "v", "m", "f", "w")
  
  charged_ <- sum(sequence %in% charged)
  hydrophobic_ <- sum(sequence %in% hydrophobic)
  
  if (hydrophobic_ == 0) {
    f4_amphipathicity = 0
  } else {
    f4_amphipathicity = charged_ / hydrophobic_
  }
  
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
    f4_amphipathicity,
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


runCSAMPPred <- function(train_path, test_path, output_path) {
  
  train_features <- create_features(train_path)
  test_features <- create_features(test_path)
  
  train_target <- create_target(train_path)
  test_target <- create_target(test_path)
  
  pca <- prcomp(train_features)
  
  train_pca <- predict(pca, train_features)
  test_pca <- predict(pca, test_features)
  
  pvals <- c()
  for (i in 1:ncol(train_pca)) {
    p_val <- wilcox.test(train_pca[train_target == 1, i], train_pca[train_target == 0, i], alternative="two.sided")$p.value
    pvals <- c(pvals, p_val)
  }
  
  X_train <- train_pca[, pvals < 0.05]
  X_test <- test_pca[, pvals < 0.05]
  
  #svm_linear <- svm(X_train, train_target, kernel = "linear")
  # svm_poly <- svm(X_train, train_target, kernel = "polynomial")
  svm_radial <- svm(X_train, train_target, kernel = "radial")
  
  predictions <- as.character(predict(svm_radial, X_test))
  y_test <- as.character(test_target)
  test_seqnames <- rownames(test_features)
  
  out <- data.frame(cbind(test_seqnames, y_test, predictions, probability = NA))
  colnames(out) <- c("ID", "target", "prediction", "probability")
  
  write.csv(out, file = output_path, row.names = FALSE, quote = FALSE)
  
  TRUE
}

runCSAMPPred(args[1], args[2], args[3])

