#!/usr/bin/env Rscript
library(tidyverse)
library(biogram)
library(ranger)
library(tidyr)
library(stringi)
library(seqR)
#--------------------------------------------------FUNCTIONS

create_mer_df <- function(seq) 
  do.call(rbind, lapply(1L:nrow(seq), function(i) {
    seq2ngrams(seq[i, ][!is.na(seq[i, ])], 5, a()[-1]) %>% 
      decode_ngrams() %>% 
      unname() %>% 
      data.frame(seq = .,
                 source_peptide = rownames(seq)[i],
                 stringsAsFactors = FALSE)
  }))



count_ampgrams <- function(mer_df, k_vector, gap_list) {
  
  sapply(mer_df[["seq"]], function(i) strsplit(i, "")[[1]], simplify = FALSE, USE.NAMES = FALSE) %>% 
    count_multimers(k_vector = k_vector,
                    kmer_gaps_list = gap_list,
                    alphabet = toupper(colnames(aaprop)),
                    with_kmer_counts = FALSE)
}

count_longest <- function(x) {
  splitted_x <- strsplit(x = paste0(as.numeric(x > 0.5), collapse = ""),
                         split = "0")[[1]]
  len <- unname(sapply(splitted_x, nchar))
  if (length(len[len > 0]) == 0) {
    0 } else {
      len[len > 0]
    }
}

calculate_statistics <- function(pred_mers) {
  (if("fold" %in% colnames(pred_mers)) {
    group_by(pred_mers, source_peptide, target, fold)
  } else {  
    group_by(pred_mers, source_peptide, target)
  }) %>% 
    summarise(fraction_true = mean(pred > 0.5),
              pred_mean = mean(pred),
              pred_median = median(pred),
              n_peptide = length(pred),
              n_pos = sum(pred > 0.5),
              pred_min = min(pred),
              pred_max = max(pred), 
              longest_pos = max(count_longest(pred)),
              n_pos_10 = sum(count_longest(pred) >= 10),
              frac_0_0.2 = sum(pred <= 0.2)/n(),
              frac_0.2_0.4 = sum(pred > 0.2 & pred <= 0.4)/n(),
              frac_0.4_0.6 = sum(pred > 0.4 & pred <= 0.6)/n(),
              frac_0.6_0.8 = sum(pred > 0.6 & pred <= 0.8)/n(),
              frac_0.8_1 = sum(pred > 0.8 & pred <= 1)/n()) %>% 
    ungroup() %>% 
    mutate(target = factor(target))
}


train_model_peptides <- function(mer_statistics) {
  train_dat <- mer_statistics %>% 
    select(c("target", "fraction_true", "pred_mean", "pred_median",
             "n_peptide", "n_pos", "pred_min", "pred_max", "longest_pos",
             "n_pos_10", "frac_0_0.2", "frac_0.2_0.4", "frac_0.4_0.6",
             "frac_0.6_0.8", "frac_0.8_1"))
  model_cv <- ranger(dependent.variable.name = "target", data = train_dat, 
                     write.forest = TRUE, probability = TRUE, num.trees = 500, 
                     verbose = FALSE, classification = TRUE, seed = 990)
  model_cv
}

extract_imp_ngrams <- function(features, binary_ngrams) {
  ft <- colnames(binary_ngrams)
  ft[as.numeric(gsub("feature", "", features))]
} 

args = commandArgs(trailingOnly=TRUE)

train_file <- args[1]
test_file <- args[2]
output_file <- args[3]

training_data <- read_fasta(train_file) %>% 
  list2matrix() %>% 
  create_mer_df()

binary_ngrams <-  count_ampgrams(training_data, 
                                 k_vector = c(1, rep(2, 4), c(rep(3, 4))),
                                 gap_list = list(NULL, NULL, 1, 2, 3, c(0, 0), c(0, 1), c(1, 0), c(1, 1)))

train_dat <- training_data %>% 
  mutate(target = as.numeric((stringi::stri_match_last_regex(source_peptide, "(?<=AMP\\=)0|1") == "1"))) %>% 
  mutate(target = ifelse(target == 1, TRUE, FALSE))


test_bis <- test_features(train_dat[["target"]], binary_ngrams, occurrences = FALSE)

imp_bigrams <- cut(test_bis, breaks = c(0, 0.05, 1))[[1]] %>% 
  extract_imp_ngrams(binary_ngrams)

# ranger_train_data <- cbind(as.matrix(binary_ngrams[, imp_bigrams]),
#                            tar = as.factor(train_dat[["target"]]))

model_cv <- ranger(x = binary_ngrams[, imp_bigrams], y = as.factor(train_dat[["target"]]),
                   write.forest = TRUE, probability = TRUE, num.trees = 2000, 
                   verbose = FALSE, seed = 990, save.memory = TRUE)

preds <- lapply(1:10, function(i) {
  n <- nrow(binary_ngrams)%/%10
  b <- seq(0, nrow(binary_ngrams), n)
  if(i < 10) {
    predict(model_cv, binary_ngrams[(b[i]+1):(b[i]+n), imp_bigrams])[["predictions"]][, "TRUE"]
  } else {
    predict(model_cv, binary_ngrams[(b[10]+1):nrow(binary_ngrams), imp_bigrams])[["predictions"]][, "TRUE"]
  }
}) %>% unlist()

mer_df <- train_dat %>% 
  mutate(pred = preds) %>% 
  select(c("source_peptide","target", "pred"))

mer_statistics <- calculate_statistics(mer_df)

peptide_model <- train_model_peptides(mer_statistics)


#------------------------test

test_seqs <- read_fasta(test_file) 

test_data <- test_seqs[which(!(grepl("AmPEP|ampir-precursor", names(test_seqs))))] %>% 
  list2matrix() %>% 
  create_mer_df()

ngrams_T <- count_ampgrams(test_data, 
                           k_vector = c(1, rep(2, 4), c(rep(3, 4))),
                           gap_list = list(NULL, NULL, 1, 2, 3, c(0, 0), c(0, 1), c(1, 0), c(1, 1)))
to_add <- imp_bigrams[which(!(imp_bigrams %in% colnames(ngrams_T)))]
ngrams_T <- cbind(ngrams_T, matrix(0, ncol = length(to_add), nrow = nrow(ngrams_T), dimnames = list(NULL, to_add)))

test_dat <- test_data %>% 
  mutate(target = as.numeric((stringi::stri_match_last_regex(source_peptide, "(?<=AMP\\=)0|1") == "1"))) %>% 
  mutate(target = ifelse(target == 1,1,0))

preds_T <- lapply(1:10, function(i) {
  n <- nrow(ngrams_T)%/%10
  b <- seq(0, nrow(ngrams_T), n)
  if(i < 10) {
    predict(model_cv, ngrams_T[(b[i]+1):(b[i]+n), imp_bigrams])[["predictions"]][, "TRUE"]
  } else {
    predict(model_cv, ngrams_T[(b[10]+1):nrow(ngrams_T), imp_bigrams])[["predictions"]][, "TRUE"]
  }
}) %>% unlist()

mer_df_T <- test_dat %>% 
  mutate(pred = preds_T) %>% 
  select(c("source_peptide", "target", "pred"))

mer_statistics_T <- calculate_statistics(mer_df_T)

res <- predict(peptide_model, mer_statistics_T)[["predictions"]][, "TRUE"]

mer_statistics_T %>% 
  select(c("source_peptide", "target")) %>% 
  mutate(probability = res) %>% 
  rename("ID" = "source_peptide") %>% 
  mutate(prediction = ifelse(probability > 0.5, 1, 0)) %>% 
  write.csv(file = output_file, row.names = FALSE)


