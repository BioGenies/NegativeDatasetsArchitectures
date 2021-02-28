#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(dplyr)
library(stringi)
library(protr)
if (!require(tidysq)) devtools::install_github("BioGenies/tidysq")
if (!require(fknn)) devtools::install_github("DominikRafacz/fknn")

# properties values used in PseACC are taken from www.csbio.sjtu.edu.cn/bioinf/PseAAC/ParaValue.htm
# properties selected according to the article
properties_names <- c("w_Hydrophobicity", "w_Mass", "w_pK1", "w_pK2", "w_pI")
custom_properties <- read.csv("models/iAMP-2L/PseAA.csv") %>%
  select(-Hydrophilicity.b) %>%
  tibble::column_to_rownames("Amino.acid") %>%
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  rename(AccNo = rowname) %>%
  mutate(AccNo = properties_names) %>%
  select(AccNo, A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P, S, T, W, Y, V)

prepare_features <- function(input_file) {
  sequences_df <- read_fasta(input_file, alphabet = "ami_bsc")
  
  sequences_df %>%
    pull(sq) %>%
    sqapply(protr::extractAPAAC, 
            lambda = 4, 
            w = 0.1, 
            props = properties_names,
            customprops = custom_properties,
            single_string = TRUE) %>%
    do.call(rbind, .) %>%
    as.data.frame() %>%
    bind_cols(sequences_df, .) %>%
    mutate(positive = as.numeric((stringi::stri_match_last_regex(name, "(?<=AMP\\=)0|1") == "1")),
           negative = 1 - positive)
}

input_df_train <- prepare_features(args[1])
input_df_test <- prepare_features(args[2])

results <- fknn(
  input_df_train %>% select(-sq, -name, -positive, -negative) %>% as.matrix,
  input_df_train %>% select(positive, negative) %>% as.matrix,
  input_df_test %>% select(-sq, -name, -positive, -negative) %>% as.matrix,
  k = 19,
  m = 1.8)

input_df_test %>%
  select(ID = name, target = positive) %>%
  mutate(prediction = round(results[, 1]),
         probability = results[, 1]) %>%
  write.csv(args[3], row.names = FALSE)