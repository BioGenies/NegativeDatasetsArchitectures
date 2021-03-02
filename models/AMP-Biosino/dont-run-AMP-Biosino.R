#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(dplyr)
library(class)
if (!require(tidysq)) devtools::install_github("BioGenies/tidysq")

polarity_values <- c(
  A = -0.591, C = 1.343, D = 1.05, E = 1.357, F = -1.006,
  G = -0.384, H = 0.336, I = -1.239, K = 1.831, L = -1.019,
  M = -0.663, N = 0.945, P = 0.189, Q = 0.931, R = 1.538,
  S = -0.228, T = -0.032, V = -1.337, W = -0.595, Y = 0.26
)

secondary_struct_values <- c(
  A = -1.302, C = 0.465, D = 0.302, E = -1.435, F = -0.59,
  G = 1.652, H = -0.417, I = -0.547, K = -0.561, L = -0.987,
  M = -1.524, N = 0.828, P = 2.081, Q = -0.179, R = 0.055,
  S = 1.399, T = 0.326, V = -0.279, W = 0.009, Y = 0.83
)

molecular_volume_values <- c(
  A = -0.733, C = -0.862, D = -3.656, E = 1.477, F = 1.891,
  G = 1.33, H = -1.673, I = 2.131, K = 0.533, L = -1.505,
  M = 2.219, N = 1.299, P = -1.628, Q = -3.005, R = 1.502,
  S = -4.76, T = 2.213, V = -0.5444, W = 0.672, Y = 3.097
)

codon_diversity_values <- c(
  A = 1.57, C = -1.02, D = -0.259, E = 0.113, F = -0.397,
  G = 1.045, H = -1.474, I = 0.393, K = -0.277, L = 1.266,
  M = -1.005, N = -0.169, P = 0.421, Q = -0.503, R = 0.44,
  S = 0.67, T = 0.908, V = 1.242, W = -2.128, Y = -0.838
)

electrostatic_values <- c(
  A = -0.146, C = -0.255, D = -3.242, E = -0.837, F = 0.412,
  G = 2.064, H = -0.078, I = 0.816, K = 1.648, L = -0.912,
  M = 1.212, N = 0.933, P = -1.392, Q = -1.853, R = 2.897,
  S = -2.647, T = 1.313, V = -1.262, W = -0.184, Y = 1.512
)

correlation_factor <- function(sequence, lambda) {
  seq_length <- length(sequence)
  mean((sequence[(1 + lambda):seq_length] -
          sequence[1:(seq_length - lambda)])^2)
}

AAC <- function(x) {
  bind_rows(sqapply(x, function(sequence) {
    ami_bsc_letters <- get_standard_alphabet("ami_bsc")[1:20]
    setNames(vapply(ami_bsc_letters, function(letter) {
      sum(letter == sequence)/length(sequence)
    }, numeric(1)), ami_bsc_letters)
  }))
}

PseAAC <- function(x,
                   feature_values,
                   feature_name,
                   lambda = 50,
                   weight_factor = 0.15) {
  sq_features <- sqapply(x, function(sequence) {
    feature_values[sequence]
  })
  ret <- bind_rows(lapply(sq_features, function(sequence) {
    setNames(vapply(1:lambda, function(lambda) {
      correlation_factor(sequence, lambda)
    }, numeric(1)), paste0(feature_name, "_", 1:lambda))
  }))
  
  # TODO: if allowing <=50 length sequences, remove NA for rowSums
  bind_cols(AAC(x), ret) %>%
    mutate(row_sum = rowSums(across())) %>%
    mutate(ret, across(.fns = ~(.x / row_sum))) %>%
    select(-row_sum)
}

# Seems like a function specific to this paper
# Basically standardizes feature values
feature_conversion <- function(feature) {
  centered_feature <- feature - mean(feature)
  centered_feature/sqrt(mean(centered_feature^2))
}

prepare_features <- function(sequence_file) {
  sequences_tbl <- read_fasta(sequence_file) %>%
    mutate(sq = remove_ambiguous(sq)) %>%
    filter(get_sq_lengths(sq) > 50)
  
  polarity_tbl <-
    PseAAC(sequences_tbl[["sq"]], polarity_values, "polarity")
  secondary_struct_tbl <-
    PseAAC(sequences_tbl[["sq"]], secondary_struct_values, "secondary_structure") %>%
    select(starts_with("secondary_structure"))
  molecular_volume_tbl <-
    PseAAC(sequences_tbl[["sq"]], molecular_volume_values, "molecular_volume") %>%
    select(starts_with("molecular_volume"))
  codon_diversity_tbl <-
    PseAAC(sequences_tbl[["sq"]], codon_diversity_values, "codon_diversity") %>%
    select(starts_with("codon_diversity"))
  electrostatic_tbl <-
    PseAAC(sequences_tbl[["sq"]], electrostatic_values, "electrostatic") %>%
    select(starts_with("electrostatic"))
  
  # TODO: select only the 25 relevant features
  bind_cols(id = sequences_tbl[["name"]],
            polarity_tbl,
            secondary_struct_tbl,
            molecular_volume_tbl,
            codon_diversity_tbl,
            electrostatic_tbl) %>%
    select(everything()) %>%
    mutate(target = as.factor(ifelse(grepl("AMP=1", id), 1, 0)))
}

train_df <- prepare_features(args[1])
test_df <- prepare_features(args[2])

# knn1 doesn't really allow for probabilities, so I made them
#  either 1 or 0, depending on predicted class
preds <- knn1(select(train_df, -c(id, target)),
              select(test_df, -c(id, target)),
              pull(train_df, target)) %>%
  as_tibble() %>%
  transmute(prediction = as.character(value),
            probability = as.numeric(prediction))

res <- test_df %>%
  select(id, target) %>%
  bind_cols(preds)

write.csv(res, args[3], row.names = FALSE)
