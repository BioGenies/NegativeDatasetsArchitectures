#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(dplyr)
library(class)
library(checkmate)
library(tidysq)
library(protr)

lambda <- 50

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

custom_prop_names <- c("polarity", "secondary_structure", "molecular_volume",
                       "codon_diversity", "electrostatic")

custom_pseAAC_props <- rbind(polarity_values,
                             secondary_struct_values,
                             molecular_volume_values,
                             codon_diversity_values,
                             electrostatic_values) %>%
  as_tibble() %>%
  mutate(AccNo = custom_prop_names) %>%
  select(AccNo, A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P ,S, T, W, Y, V)

feature_names <- c(
  "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
  "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
  paste0(rep(custom_prop_names, each = lambda), "_", 1:50)
)


prepare_features <- function(sequence_file) {
  sequences_tbl <- read_fasta(sequence_file) %>%
    mutate(sq = remove_ambiguous(sq)) %>%
    filter(get_sq_lengths(sq) > lambda)
  
  ret <- sqapply(sequences_tbl[["sq"]], function(sequence) {
    ret <- c(
      extractPAAC(sequence,
                  lambda = lambda,
                  props = "polarity",
                  customprops = custom_pseAAC_props),
      extractPAAC(sequence,
                  lambda = lambda,
                  props = "secondary_structure",
                  customprops = custom_pseAAC_props)[21:70],
      extractPAAC(sequence,
                  lambda = lambda,
                  props = "molecular_volume",
                  customprops = custom_pseAAC_props)[21:70],
      extractPAAC(sequence,
                  lambda = lambda,
                  props = "codon_diversity",
                  customprops = custom_pseAAC_props)[21:70],
      extractPAAC(sequence,
                  lambda = lambda,
                  props = "electrostatic",
                  customprops = custom_pseAAC_props)[21:70]
    )
    names(ret) <- feature_names
    ret
  }, single_string = TRUE)
  
  # TODO: select only the 25 relevant features
  bind_rows(ret) %>%
    mutate(id = sequences_tbl[["name"]]) %>%
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
